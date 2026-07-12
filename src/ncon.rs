use ndarray::{ArrayD, IxDyn};

pub type NconResult = Result<ArrayD<f64>, String>;

pub fn ncon_tensors(
    a: &ArrayD<f64>,
    b: &ArrayD<f64>,
    labels_a: &[isize],
    labels_b: &[isize],
) -> NconResult {
    let ndim_a = a.ndim();
    let ndim_b = b.ndim();

    if labels_a.len() != ndim_a {
        return Err(format!(
            "labels_a length {} != tensor a ndim {}", labels_a.len(), ndim_a
        ));
    }
    if labels_b.len() != ndim_b {
        return Err(format!(
            "labels_b length {} != tensor b ndim {}", labels_b.len(), ndim_b
        ));
    }

    let a_shape = a.shape();
    let b_shape = b.shape();

    // Check for duplicate positive labels on a
    {
        let mut seen = Vec::new();
        for &lbl in labels_a.iter().filter(|&&l| l > 0) {
            if seen.contains(&lbl) {
                return Err(format!("Duplicate contracted label {} on tensor A", lbl));
            }
            seen.push(lbl);
        }
    }
    // Check for duplicate positive labels on b
    {
        let mut seen = Vec::new();
        for &lbl in labels_b.iter().filter(|&&l| l > 0) {
            if seen.contains(&lbl) {
                return Err(format!("Duplicate contracted label {} on tensor B", lbl));
            }
            seen.push(lbl);
        }
    }

    let mut a_contracted: Vec<usize> = Vec::new();
    let mut a_free: Vec<usize> = Vec::new();
    let mut b_free: Vec<usize> = Vec::new();

    for (i, &lbl) in labels_a.iter().enumerate() {
        if lbl > 0 {
            a_contracted.push(i);
        } else {
            a_free.push(i);
        }
    }
    for (i, &lbl) in labels_b.iter().enumerate() {
        if lbl > 0 {
            // will match with a's contracted
            let _ = (i, lbl);
        } else {
            b_free.push(i);
        }
    }

    let mut b_contracted: Vec<usize> = Vec::new();
    for &ax_a in &a_contracted {
        let target = labels_a[ax_a];
        match labels_b.iter().position(|&x| x == target) {
            Some(ax_b) => {
                let dim_a = a_shape[ax_a];
                let dim_b = b_shape[ax_b];
                if dim_a != dim_b {
                    return Err(format!(
                        "Contracted dimension mismatch for label {}: a[{}]={}, b[{}]={}",
                        target, ax_a, dim_a, ax_b, dim_b
                    ));
                }
                b_contracted.push(ax_b);
            }
            None => {
                return Err(format!(
                    "Contracted label {} on tensor A has no match on tensor B", target
                ));
            }
        }
    }

    // Check for orphan contracted labels on b
    for &lbl in labels_b.iter().filter(|&&l| l > 0) {
        if !labels_a.contains(&lbl) {
            return Err(format!(
                "Contracted label {} on tensor B has no match on tensor A", lbl
            ));
        }
    }

    let perm_a: Vec<usize> = a_free.iter().copied()
        .chain(a_contracted.iter().copied())
        .collect();
    let a_perm = a.clone().permuted_axes(perm_a.clone())
        .as_standard_layout().to_owned();

    let perm_b: Vec<usize> = b_contracted.iter().copied()
        .chain(b_free.iter().copied())
        .collect();
    let b_perm = b.clone().permuted_axes(perm_b)
        .as_standard_layout().to_owned();

    let free_shape_a: Vec<usize> = a_free.iter().map(|&i| a_shape[i]).collect();
    let free_shape_b: Vec<usize> = b_free.iter().map(|&i| b_shape[i]).collect();
    let contracted_size: usize = a_contracted.iter().map(|&i| a_shape[i]).product();

    let a_rows = free_shape_a.iter().product::<usize>();
    let a_cols = contracted_size;
    let b_rows = contracted_size;
    let b_cols = free_shape_b.iter().product::<usize>();

    let a_mat = crate::backend::NdArrayBackend::as_2d(&a_perm, a_rows, a_cols);
    let b_mat = crate::backend::NdArrayBackend::as_2d(&b_perm, b_rows, b_cols);

    let result_mat = crate::backend::NdArrayBackend::matmul_2d(&a_mat, &b_mat);

    let mut result_shape = free_shape_a;
    result_shape.extend_from_slice(&free_shape_b);

    result_mat
        .into_shape_with_order(IxDyn(&result_shape))
        .map_err(|e| format!("Failed to reshape result: {}", e))
}
