#[cfg(test)]
mod tests {
    use rust_tensornetwork::backend::{NdArrayBackend, TensorBackend};
    use rust_tensornetwork::network::TensorNetwork;
    use rust_tensornetwork::ops;

    type Net = TensorNetwork<NdArrayBackend>;
    type BT = NdArrayBackend;

    #[test]
    fn test_create_node() {
        let mut net = Net::new();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = BT::convert_to_tensor(&data, &[2, 2]);
        let nid = net.add_node_with_edges(tensor, "A");
        assert_eq!(net.node_count(), 1);
        let node = net.get_node(nid);
        assert_eq!(node.rank(), 2);
        assert!(node.active);
    }

    #[test]
    fn test_connect_and_contract_vector_dot() {
        let mut net = Net::new();
        let a = net.add_node_with_edges(BT::convert_to_tensor(&[1.0_f64; 10], &[10]), "a");
        let b = net.add_node_with_edges(BT::convert_to_tensor(&[2.0_f64; 10], &[10]), "b");

        let ea = net.get_node(a).edges[0];
        let eb = net.get_node(b).edges[0];
        let connected = ops::connect(ea, eb, &mut net, Some("ab")).unwrap();

        let result_id = ops::contract(connected, &mut net, Some("dot")).unwrap();
        let result = net.get_node(result_id);
        let tensor_value = result.tensor.first().copied().unwrap_or(0.0);
        assert!((tensor_value - 20.0).abs() < 1e-10);

        // Old nodes are inactive after contraction
        assert!(!net.nodes[a].active);
        assert!(!net.nodes[b].active);
    }

    #[test]
    fn test_matrix_vector_contraction() {
        let mut net = Net::new();
        let mat = BT::convert_to_tensor(&[1.0_f64; 12], &[3, 4]);
        let vec = BT::convert_to_tensor(&[2.0_f64; 4], &[4]);
        let m = net.add_node_with_edges(mat, "M");
        let v = net.add_node_with_edges(vec, "v");

        let e_m = net.get_node(m).edges[1];
        let e_v = net.get_node(v).edges[0];
        let con = ops::connect(e_m, e_v, &mut net, Some("mv")).unwrap();

        let result_id = ops::contract(con, &mut net, Some("mv_result")).unwrap();
        let result = net.get_node(result_id);
        assert_eq!(BT::shape(&result.tensor), vec![3]);

        assert!(!net.nodes[m].active);
        assert!(!net.nodes[v].active);
    }

    #[test]
    fn test_contract_between() {
        let mut net = Net::new();
        let m = net.add_node_with_edges(BT::convert_to_tensor(&[1.0_f64; 12], &[3, 4]), "M");
        let v = net.add_node_with_edges(BT::convert_to_tensor(&[2.0_f64; 4], &[4]), "v");

        let e_m = net.get_node(m).edges[1];
        let e_v = net.get_node(v).edges[0];
        ops::connect(e_m, e_v, &mut net, Some("mv")).unwrap();

        let result_id = ops::contract_between(m, v, &mut net, false, Some("mv_r")).unwrap();
        let result = net.get_node(result_id);
        assert_eq!(BT::shape(&result.tensor), vec![3]);
    }

    #[test]
    fn test_trace() {
        let mut net = Net::new();
        let mat = BT::convert_to_tensor(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let node_id = net.add_node_with_edges(mat, "I");

        let e0 = net.get_node(node_id).edges[0];
        let e1 = net.get_node(node_id).edges[1];
        let trace_edge = ops::connect(e0, e1, &mut net, Some("trace")).unwrap();

        let result_id = ops::contract(trace_edge, &mut net, Some("trace_result")).unwrap();
        let result = net.get_node(result_id);
        let trace_val = result.tensor.first().copied().unwrap_or(0.0);
        assert!((trace_val - 2.0).abs() < 1e-10);
        // trace of 2x2 matrix is a scalar (0-dim tensor with 1 element)
        assert_eq!(BT::ndim(&result.tensor), 0);
    }

    #[test]
    fn test_trace_axis_names() {
        let mut net = Net::new();
        // symmetric [3, 3] matrix — axes must have equal dim to trace
        let mat = BT::convert_to_tensor(&[1.0_f64; 9], &[3, 3]);
        let node_id = net.add_node(mat, "M_3x3", Some(vec!["a".into(), "b".into()]));

        let e0 = net.get_node(node_id).edges[0];
        let e1 = net.get_node(node_id).edges[1];
        let trace_edge = ops::connect(e0, e1, &mut net, Some("tr")).unwrap();

        let result_id = ops::contract(trace_edge, &mut net, Some("tr_result")).unwrap();
        let result = net.get_node(result_id);

        assert_eq!(BT::ndim(&result.tensor), 0);
        assert!(result.axis_names.is_empty(),
            "axis_names should be empty after full trace, got {:?}", result.axis_names);
    }

    #[test]
    fn test_trace_partial_axis_names() {
        let mut net = Net::new();
        // Shape [2, 3, 3, 4], trace axes 1 and 2 → result shape [2, 4]
        let t = BT::convert_to_tensor(&[1.0_f64; 72], &[2, 3, 3, 4]);
        let node_id = net.add_node(t, "T", Some(vec!["x".into(), "y1".into(), "y2".into(), "z".into()]));

        let e1 = net.get_node(node_id).edges[1];
        let e2 = net.get_node(node_id).edges[2];
        let trace_edge = ops::connect(e1, e2, &mut net, Some("tr")).unwrap();

        let result_id = ops::contract(trace_edge, &mut net, Some("partial_trace")).unwrap();
        let result = net.get_node(result_id);

        assert_eq!(BT::shape(&result.tensor), vec![2, 4],
            "partial trace of [2,3,3,4] over axes 1,2 should give [2,4]");
        assert_eq!(result.axis_names, vec!["x", "z"],
            "axis names after partial trace should be [x, z], got {:?}", result.axis_names);
    }

    #[test]
    fn test_outer_product() {
        let mut net = Net::new();
        let a = net.add_node_with_edges(BT::convert_to_tensor(&[1.0, 2.0], &[2]), "a");
        let b = net.add_node_with_edges(BT::convert_to_tensor(&[3.0, 4.0], &[2]), "b");

        let result_id = ops::outer_product(a, b, &mut net, Some("outer")).unwrap();
        let result = net.get_node(result_id);
        assert_eq!(BT::shape(&result.tensor), vec![2, 2]);
    }

    #[test]
    fn test_multi_edge_contract_between() {
        let mut net = Net::new();
        let t1 = BT::convert_to_tensor(&[1.0_f64; 24], &[2, 3, 4]);
        let t2 = BT::convert_to_tensor(&[2.0_f64; 30], &[2, 3, 5]);
        let n1 = net.add_node_with_edges(t1, "n1");
        let n2 = net.add_node_with_edges(t2, "n2");

        let e1a = net.get_node(n1).edges[0];
        let e2a = net.get_node(n2).edges[0];
        ops::connect(e1a, e2a, &mut net, Some("c0")).unwrap();

        let e1b = net.get_node(n1).edges[1];
        let e2b = net.get_node(n2).edges[1];
        ops::connect(e1b, e2b, &mut net, Some("c1")).unwrap();

        let result_id = ops::contract_between(n1, n2, &mut net, false, Some("r")).unwrap();
        let result = net.get_node(result_id);
        assert_eq!(BT::shape(&result.tensor), vec![4, 5]);
    }

    #[test]
    fn test_inactive_node_panics() {
        let mut net = Net::new();
        let a = net.add_node_with_edges(BT::convert_to_tensor(&[1.0_f64; 10], &[10]), "a");
        let b = net.add_node_with_edges(BT::convert_to_tensor(&[2.0_f64; 10], &[10]), "b");

        let ea = net.get_node(a).edges[0];
        let eb = net.get_node(b).edges[0];
        let con = ops::connect(ea, eb, &mut net, Some("ab")).unwrap();
        let result_id = ops::contract(con, &mut net, Some("r")).unwrap();

        // a and b are now inactive — accessing them via get_node should panic
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            net.get_node(a);
        }));
        assert!(result.is_err(), "get_node on inactive node should panic");

        // result node should still be active and accessible
        let r_node = net.get_node(result_id);
        assert!(r_node.active);
    }

    #[test]
    fn test_connect_deactivates_old_dangling_edges() {
        let mut net = Net::new();
        let a = net.add_node_with_edges(BT::convert_to_tensor(&[1.0_f64; 10], &[10]), "a");
        let b = net.add_node_with_edges(BT::convert_to_tensor(&[2.0_f64; 10], &[10]), "b");

        let edge_count_before = net.edge_count();
        let ea = net.get_node(a).edges[0];
        let eb = net.get_node(b).edges[0];

        ops::connect(ea, eb, &mut net, Some("ab")).unwrap();

        // Old dangling edges are deactivated, not removed from slab
        assert!(!net.edges[ea].active, "old dangling edge ea should be inactive");
        assert!(!net.edges[eb].active, "old dangling edge eb should be inactive");

        // Edge count should decrease by 2 (2 deactivated, 1 new active)
        assert_eq!(net.edge_count(), edge_count_before - 1);
    }

    #[test]
    fn test_connect_self_panics() {
        let mut net = Net::new();
        let a = net.add_node_with_edges(BT::convert_to_tensor(&[1.0_f64; 10], &[10]), "a");
        let ea = net.get_node(a).edges[0];

        let result = ops::connect(ea, ea, &mut net, Some("self"));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Cannot connect an edge to itself");
    }

    #[test]
    fn test_connect_inactive_edge_returns_err() {
        let mut net = Net::new();
        let a = net.add_node_with_edges(BT::convert_to_tensor(&[1.0_f64; 10], &[10]), "a");
        let b = net.add_node_with_edges(BT::convert_to_tensor(&[2.0_f64; 10], &[10]), "b");

        let ea = net.get_node(a).edges[0];
        let eb = net.get_node(b).edges[0];
        net.deactivate_edge(ea);

        let result = ops::connect(ea, eb, &mut net, Some("ab"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not an active dangling edge"));
    }

    #[test]
    fn test_node_count_excludes_inactive() {
        let mut net = Net::new();
        let a = net.add_node_with_edges(BT::convert_to_tensor(&[1.0_f64; 10], &[10]), "a");
        let b = net.add_node_with_edges(BT::convert_to_tensor(&[2.0_f64; 10], &[10]), "b");

        assert_eq!(net.node_count(), 2);

        let ea = net.get_node(a).edges[0];
        let eb = net.get_node(b).edges[0];
        let con = ops::connect(ea, eb, &mut net, Some("ab")).unwrap();
        ops::contract(con, &mut net, Some("r")).unwrap();

        // 2 inactive + 1 active = 3 total in slab, but only 1 active
        assert_eq!(net.nodes.len(), 3); // slab still has all 3
        assert_eq!(net.node_count(), 1); // but only 1 active
    }
}
