use slab::Slab;
use crate::backend::TensorBackend;

pub type NodeId = usize;
pub type EdgeId = usize;

#[derive(Clone)]
pub struct Node<B: TensorBackend> {
    pub tensor: B::Tensor,
    pub edges: Vec<EdgeId>,
    pub name: String,
    pub axis_names: Vec<String>,
    pub active: bool,
}

impl<B: TensorBackend> Node<B> {
    pub fn rank(&self) -> usize { self.edges.len() }

    pub fn get_dimension(&self, axis: usize) -> Option<usize> {
        B::shape(&self.tensor).get(axis).copied()
    }
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub node1: NodeId,
    pub node2: Option<NodeId>,
    pub axis1: usize,
    pub axis2: Option<usize>,
    pub name: String,
    pub active: bool,
}

impl Edge {
    pub fn new_dangling(node: NodeId, axis: usize, name: String) -> Self {
        Edge { node1: node, node2: None, axis1: axis, axis2: None, name, active: true }
    }

    pub fn new_standard(n1: NodeId, a1: usize, n2: NodeId, a2: usize, name: String) -> Self {
        Edge { node1: n1, node2: Some(n2), axis1: a1, axis2: Some(a2), name, active: true }
    }

    pub fn is_dangling(&self) -> bool { self.node2.is_none() }

    pub fn is_trace(&self) -> bool { self.node2 == Some(self.node1) }
}

pub struct TensorNetwork<B: TensorBackend> {
    pub nodes: Slab<Node<B>>,
    pub edges: Slab<Edge>,
}

impl<B: TensorBackend> Default for TensorNetwork<B> {
    fn default() -> Self { Self::new() }
}

impl<B: TensorBackend> TensorNetwork<B> {
    pub fn new() -> Self {
        TensorNetwork { nodes: Slab::new(), edges: Slab::new() }
    }

    pub fn add_node(&mut self, tensor: B::Tensor, name: &str, axis_names: Option<Vec<String>>) -> NodeId {
        let shape = B::shape(&tensor);
        let axis_names = axis_names.unwrap_or_else(|| {
            shape.iter().enumerate().map(|(i, _)| format!("axis_{}", i)).collect()
        });

        let rank = shape.len();
        let node = Node {
            tensor, edges: Vec::with_capacity(rank),
            name: name.to_string(), axis_names, active: true,
        };

        let node_id = self.nodes.insert(node);
        for a in 0..rank {
            let edge = Edge::new_dangling(node_id, a, format!("dangling_{}", a));
            let edge_id = self.edges.insert(edge);
            self.nodes[node_id].edges.push(edge_id);
        }
        node_id
    }

    pub fn add_node_with_edges(&mut self, tensor: B::Tensor, name: &str) -> NodeId {
        self.add_node(tensor, name, None)
    }

    pub fn deactivate_node(&mut self, node_id: NodeId) {
        self.nodes[node_id].active = false;
    }

    pub fn deactivate_edge(&mut self, edge_id: EdgeId) {
        self.edges[edge_id].active = false;
    }

    pub fn get_node(&self, id: NodeId) -> &Node<B> {
        let node = &self.nodes[id];
        assert!(node.active, "Node {} is inactive", id);
        node
    }

    pub fn get_node_mut(&mut self, id: NodeId) -> &mut Node<B> {
        let node = &mut self.nodes[id];
        assert!(node.active, "Node {} is inactive", id);
        node
    }

    pub fn get_edge(&self, id: EdgeId) -> &Edge {
        let edge = &self.edges[id];
        assert!(edge.active, "Edge {} is inactive", id);
        edge
    }

    pub fn get_edge_mut(&mut self, id: EdgeId) -> &mut Edge {
        let edge = &mut self.edges[id];
        assert!(edge.active, "Edge {} is inactive", id);
        edge
    }

    /// Read an edge without checking active flag (used internally during transfer).
    pub fn get_edge_raw(&self, id: EdgeId) -> &Edge { &self.edges[id] }

    pub fn node_count(&self) -> usize {
        self.nodes.iter().filter(|(_, n)| n.active).count()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.iter().filter(|(_, e)| e.active).count()
    }
}
