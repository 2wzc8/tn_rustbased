use crate::backend::TensorBackend;
use crate::network::{TensorNetwork, NodeId, EdgeId, Edge};

pub fn connect<B: TensorBackend>(
    edge1_id: EdgeId,
    edge2_id: EdgeId,
    network: &mut TensorNetwork<B>,
    name: Option<&str>,
) -> Result<EdgeId, String> {
    if edge1_id == edge2_id {
        return Err("Cannot connect an edge to itself".to_string());
    }

    let e1 = network.get_edge_raw(edge1_id).clone();
    let e2 = network.get_edge_raw(edge2_id).clone();

    if !e1.active || !e1.is_dangling() {
        return Err(format!("Edge {} is not an active dangling edge", edge1_id));
    }
    if !e2.active || !e2.is_dangling() {
        return Err(format!("Edge {} is not an active dangling edge", edge2_id));
    }

    let dim1 = edge_dimension::<B>(&e1, network)?;
    let dim2 = edge_dimension::<B>(&e2, network)?;
    if dim1 != dim2 {
        return Err(format!(
            "Cannot connect edges of unequal dimension: {} vs {}", dim1, dim2
        ));
    }

    let n1_id = e1.node1;
    let a1 = e1.axis1;
    let n2_id = e2.node1;
    let a2 = e2.axis1;

    let edge_name = name.unwrap_or("__connected__").to_string();
    let new_edge = Edge::new_standard(n1_id, a1, n2_id, a2, edge_name);
    let new_id = network.edges.insert(new_edge);

    network.nodes[n1_id].edges[a1] = new_id;
    network.nodes[n2_id].edges[a2] = new_id;

    network.deactivate_edge(edge1_id);
    network.deactivate_edge(edge2_id);

    Ok(new_id)
}

pub fn contract<B: TensorBackend>(
    edge_id: EdgeId,
    network: &mut TensorNetwork<B>,
    name: Option<&str>,
) -> Result<NodeId, String> {
    let edge = network.get_edge_raw(edge_id).clone();
    if !edge.active || edge.is_dangling() {
        return Err(format!("Cannot contract dangling or inactive edge {}", edge_id));
    }

    if edge.node2 == Some(edge.node1) {
        return contract_trace::<B>(edge_id, network, name);
    }

    let n1_id = edge.node1;
    let n2_id = edge.node2.unwrap();
    let a1 = edge.axis1;
    let a2 = edge.axis2.unwrap();

    if !network.nodes[n1_id].active || !network.nodes[n2_id].active {
        return Err("Cannot contract: one of the nodes is inactive".to_string());
    }

    let tensor1 = network.nodes[n1_id].tensor.clone();
    let tensor2 = network.nodes[n2_id].tensor.clone();
    let new_tensor = B::tensordot(&tensor1, &tensor2, &[a1], &[a2]);

    let new_name = name.unwrap_or(&format!(
        "{}_x_{}", network.nodes[n1_id].name, network.nodes[n2_id].name
    )).to_string();
    let new_node_id = network.add_node_with_edges(new_tensor, &new_name);

    transfer_edges::<B>(network, n1_id, n2_id, new_node_id, &[edge_id]);

    Ok(new_node_id)
}

pub fn contract_trace<B: TensorBackend>(
    edge_id: EdgeId,
    network: &mut TensorNetwork<B>,
    name: Option<&str>,
) -> Result<NodeId, String> {
    let edge = network.get_edge_raw(edge_id).clone();
    if !edge.active || edge.is_dangling() {
        return Err("Cannot trace dangling or inactive edge".to_string());
    }
    if edge.node2 != Some(edge.node1) {
        return Err("Edge is not a trace edge".to_string());
    }

    let n_id = edge.node1;
    if !network.nodes[n_id].active {
        return Err("Cannot trace: node is inactive".to_string());
    }

    let a1 = edge.axis1;
    let a2 = edge.axis2.unwrap();

    let tensor = network.nodes[n_id].tensor.clone();
    let traced = B::trace(&tensor, a1, a2);

    let new_name = name.unwrap_or(&format!("trace_{}", network.nodes[n_id].name)).to_string();
    let new_node_id = network.add_node_with_edges(traced, &new_name);

    let remaining: Vec<(usize, EdgeId)> = network.nodes[n_id].edges.iter()
        .copied()
        .enumerate()
        .filter(|&(_, eid)| eid != edge_id)
        .collect();

    let traced_axes = {
        let mut v = vec![a1, a2];
        v.sort();
        v
    };
    let mut new_axis_names: Vec<String> = Vec::new();
    for (ax, name) in network.nodes[n_id].axis_names.iter().enumerate() {
        if !traced_axes.contains(&ax) {
            new_axis_names.push(name.clone());
        }
    }

    let mut new_edges = Vec::new();
    for (new_axis, &(_, eid)) in remaining.iter().enumerate() {
        let e = network.edges[eid].clone();
        let mut updated = e.clone();
        if updated.node1 == n_id {
            updated.axis1 = new_axis;
            updated.node1 = new_node_id;
        } else if updated.node2 == Some(n_id) {
            updated.axis2 = Some(new_axis);
            updated.node2 = Some(new_node_id);
        }
        network.edges[eid] = updated;
        new_edges.push(eid);
    }

    network.nodes[new_node_id].edges = new_edges;
    network.nodes[new_node_id].axis_names = new_axis_names;

    network.nodes[n_id].edges.clear();
    network.deactivate_node(n_id);
    network.deactivate_edge(edge_id);

    Ok(new_node_id)
}

pub fn contract_between<B: TensorBackend>(
    node1_id: NodeId,
    node2_id: NodeId,
    network: &mut TensorNetwork<B>,
    allow_outer_product: bool,
    name: Option<&str>,
) -> Result<NodeId, String> {
    if !network.nodes[node1_id].active || !network.nodes[node2_id].active {
        return Err("contract_between: one of the nodes is inactive".to_string());
    }

    if node1_id == node2_id {
        let trace_edges: Vec<EdgeId> = network.nodes[node1_id].edges.iter()
            .copied()
            .filter(|&eid| {
                let e = network.get_edge_raw(eid);
                e.active && e.node2 == Some(node1_id)
            })
            .collect();

        if trace_edges.is_empty() {
            return Err("No trace edges found for self-contraction".to_string());
        }
        return contract_trace::<B>(trace_edges[0], network, name);
    }

    let shared_edges = get_shared_edges(node1_id, node2_id, network);

    if shared_edges.is_empty() {
        if allow_outer_product {
            return outer_product(node1_id, node2_id, network, name);
        }
        return Err(format!(
            "No shared edges between nodes {} and {}",
            network.nodes[node1_id].name, network.nodes[node2_id].name
        ));
    }

    let mut axes1 = Vec::new();
    let mut axes2 = Vec::new();
    for &eid in &shared_edges {
        let e = network.get_edge_raw(eid);
        if e.node1 == node1_id {
            axes1.push(e.axis1);
            axes2.push(e.axis2.unwrap());
        } else {
            axes1.push(e.axis2.unwrap());
            axes2.push(e.axis1);
        }
    }

    let tensor1 = network.nodes[node1_id].tensor.clone();
    let tensor2 = network.nodes[node2_id].tensor.clone();
    let new_tensor = B::tensordot(&tensor1, &tensor2, &axes1, &axes2);

    let new_name = name.unwrap_or(&format!(
        "{}_{}", network.nodes[node1_id].name, network.nodes[node2_id].name
    )).to_string();
    let new_node_id = network.add_node_with_edges(new_tensor, &new_name);

    transfer_edges::<B>(network, node1_id, node2_id, new_node_id, &shared_edges);

    Ok(new_node_id)
}

pub fn outer_product<B: TensorBackend>(
    node1_id: NodeId,
    node2_id: NodeId,
    network: &mut TensorNetwork<B>,
    name: Option<&str>,
) -> Result<NodeId, String> {
    if !network.nodes[node1_id].active || !network.nodes[node2_id].active {
        return Err("outer_product: one of the nodes is inactive".to_string());
    }

    let t1 = network.nodes[node1_id].tensor.clone();
    let t2 = network.nodes[node2_id].tensor.clone();
    let new_tensor = B::outer_product(&t1, &t2);

    let new_name = name.unwrap_or(&format!(
        "{}_outer_{}", network.nodes[node1_id].name, network.nodes[node2_id].name
    )).to_string();
    let new_node_id = network.add_node_with_edges(new_tensor, &new_name);

    let n1_rank = network.nodes[node1_id].edges.len();
    let n1_edges: Vec<EdgeId> = network.nodes[node1_id].edges.clone();
    let n2_edges: Vec<EdgeId> = network.nodes[node2_id].edges.clone();

    for (i, &eid) in n1_edges.iter().enumerate() {
        let e = network.edges[eid].clone();
        let mut updated = e.clone();
        if updated.node1 == node1_id {
            updated.axis1 = i;
            updated.node1 = new_node_id;
        } else if updated.node2 == Some(node1_id) {
            updated.axis2 = Some(i);
            updated.node2 = Some(new_node_id);
        }
        network.edges[eid] = updated;
    }

    for (i, &eid) in n2_edges.iter().enumerate() {
        let e = network.edges[eid].clone();
        let mut updated = e.clone();
        if updated.node1 == node2_id {
            updated.axis1 = i + n1_rank;
            updated.node1 = new_node_id;
        } else if updated.node2 == Some(node2_id) {
            updated.axis2 = Some(i + n1_rank);
            updated.node2 = Some(new_node_id);
        }
        network.edges[eid] = updated;
    }

    let mut new_edges = n1_edges;
    new_edges.extend_from_slice(&n2_edges);
    network.nodes[new_node_id].edges = new_edges;

    network.nodes[node1_id].edges.clear();
    network.nodes[node2_id].edges.clear();
    network.deactivate_node(node1_id);
    network.deactivate_node(node2_id);

    Ok(new_node_id)
}

pub fn get_shared_edges<B: TensorBackend>(
    node1_id: NodeId,
    node2_id: NodeId,
    network: &TensorNetwork<B>,
) -> Vec<EdgeId> {
    let mut shared = Vec::new();
    for &eid in &network.nodes[node1_id].edges {
        let e = &network.edges[eid];
        if e.active && !e.is_dangling() {
            let nodes = (e.node1, e.node2);
            if (nodes == (node1_id, Some(node2_id))) || (nodes == (node2_id, Some(node1_id))) {
                shared.push(eid);
            }
        }
    }
    shared
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn edge_dimension<B: TensorBackend>(edge: &Edge, network: &TensorNetwork<B>) -> Result<usize, String> {
    let node = &network.nodes[edge.node1];
    let shape = B::shape(&node.tensor);
    shape.get(edge.axis1).copied().ok_or_else(|| "Invalid axis".to_string())
}

fn transfer_edges<B: TensorBackend>(
    network: &mut TensorNetwork<B>,
    node1_id: NodeId,
    node2_id: NodeId,
    new_node_id: NodeId,
    contracted_edges: &[EdgeId],
) {
    let n1_edges_raw: Vec<EdgeId> = network.nodes[node1_id].edges.clone();
    let n2_edges_raw: Vec<EdgeId> = network.nodes[node2_id].edges.clone();

    let n1_edges_kept: Vec<(usize, EdgeId)> = n1_edges_raw.iter()
        .copied()
        .enumerate()
        .filter(|(_, eid)| !contracted_edges.contains(eid))
        .collect();

    let n2_edges_kept: Vec<(usize, EdgeId)> = n2_edges_raw.iter()
        .copied()
        .enumerate()
        .filter(|(_, eid)| !contracted_edges.contains(eid))
        .collect();

    let n1_rank = n1_edges_kept.len();
    let mut new_axis_names = Vec::new();
    let mut all_edges = Vec::new();

    for (new_axis, &(old_axis, eid)) in n1_edges_kept.iter().enumerate() {
        let e = network.edges[eid].clone();
        let mut updated = e.clone();
        if updated.node1 == node1_id {
            updated.axis1 = new_axis;
            updated.node1 = new_node_id;
            new_axis_names.push(network.nodes[node1_id].axis_names[old_axis].clone());
        } else if updated.node2 == Some(node1_id) {
            updated.axis2 = Some(new_axis);
            updated.node2 = Some(new_node_id);
            new_axis_names.push(network.nodes[node1_id].axis_names[old_axis].clone());
        }
        network.edges[eid] = updated;
        all_edges.push(eid);
    }

    for (new_axis_offset, &(old_axis, eid)) in n2_edges_kept.iter().enumerate() {
        let new_axis = n1_rank + new_axis_offset;
        let e = network.edges[eid].clone();
        let mut updated = e.clone();
        if updated.node1 == node2_id {
            updated.axis1 = new_axis;
            updated.node1 = new_node_id;
            new_axis_names.push(network.nodes[node2_id].axis_names[old_axis].clone());
        } else if updated.node2 == Some(node2_id) {
            updated.axis2 = Some(new_axis);
            updated.node2 = Some(new_node_id);
            new_axis_names.push(network.nodes[node2_id].axis_names[old_axis].clone());
        }
        network.edges[eid] = updated;
        all_edges.push(eid);
    }

    network.nodes[new_node_id].edges = all_edges;
    network.nodes[new_node_id].axis_names = new_axis_names;

    network.nodes[node1_id].edges.clear();
    network.nodes[node2_id].edges.clear();
    network.deactivate_node(node1_id);
    network.deactivate_node(node2_id);

    for &eid in contracted_edges {
        network.deactivate_edge(eid);
    }
}
