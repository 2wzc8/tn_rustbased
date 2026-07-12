use ndarray::{ArrayD, IxDyn, Array, Ix2};

pub type ArrayTensor = ArrayD<f64>;

/// Abstract tensor backend.
///
/// Currently `f64` is hardcoded as the scalar type. Future versions will
/// introduce a `Scalar` trait bound (`num_traits::Num + Clone + ...`)
/// to support `Complex<f64>` and other numeric types, matching the
/// multi-backend flexibility of Google's TensorNetwork Python library.
pub trait TensorBackend {
    type Tensor: Clone;

    fn shape(tensor: &Self::Tensor) -> Vec<usize>;
    fn ndim(tensor: &Self::Tensor) -> usize { Self::shape(tensor).len() }
    fn reshape(tensor: &Self::Tensor, shape: &[usize]) -> Self::Tensor;
    fn transpose(tensor: &Self::Tensor, perm: &[usize]) -> Self::Tensor;
    fn tensordot(a: &Self::Tensor, b: &Self::Tensor, axes_a: &[usize], axes_b: &[usize]) -> Self::Tensor;
    fn outer_product(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;
    fn trace(tensor: &Self::Tensor, axis1: usize, axis2: usize) -> Self::Tensor;
    fn multiply(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;
    fn conj(tensor: &Self::Tensor) -> Self::Tensor;
    fn convert_to_tensor(data: &[f64], shape: &[usize]) -> Self::Tensor;
    fn zeros(shape: &[usize]) -> Self::Tensor;
}

pub struct NdArrayBackend;

impl NdArrayBackend {
    pub fn as_2d(a: &ArrayD<f64>, rows: usize, cols: usize) -> Array<f64, Ix2> {
        let flat = a.as_standard_layout();
        Array::from_shape_vec((rows, cols), flat.iter().copied().collect()).unwrap()
    }

    pub fn matmul_2d(a: &Array<f64, Ix2>, b: &Array<f64, Ix2>) -> ArrayD<f64> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();
        assert_eq!(k, k2, "Matrix dimension mismatch: {} vs {}", k, k2);
        let mut result_data = vec![0.0_f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for ik in 0..k {
                    sum += a[(i, ik)] * b[(ik, j)];
                }
                result_data[i * n + j] = sum;
            }
        }
        ArrayD::from_shape_vec(IxDyn(&[m, n]), result_data).unwrap()
    }
}

impl TensorBackend for NdArrayBackend {
    type Tensor = ArrayTensor;

    fn shape(tensor: &Self::Tensor) -> Vec<usize> { tensor.shape().to_vec() }

    fn reshape(tensor: &Self::Tensor, shape: &[usize]) -> Self::Tensor {
        tensor.clone().into_shape_with_order(IxDyn(shape)).unwrap()
    }

    fn transpose(tensor: &Self::Tensor, perm: &[usize]) -> Self::Tensor {
        let perm_vec: Vec<usize> = perm.to_vec();
        tensor.clone().permuted_axes(perm_vec).as_standard_layout().to_owned()
    }

    // TODO: change trait signature to return Result<Self::Tensor, ...>
    // so that backend errors propagate instead of panicking here.
    fn tensordot(a: &Self::Tensor, b: &Self::Tensor, axes_a: &[usize], axes_b: &[usize]) -> Self::Tensor {
        let ndim_a = a.ndim();
        let ndim_b = b.ndim();
        let mut labels_a: Vec<isize> = Vec::new();
        let mut labels_b: Vec<isize> = Vec::new();

        for i in 0..ndim_a {
            if axes_a.contains(&i) {
                labels_a.push(1 + axes_a.iter().position(|&x| x == i).unwrap() as isize);
            } else {
                labels_a.push(-(1 + i as isize));
            }
        }
        for i in 0..ndim_b {
            if axes_b.contains(&i) {
                labels_b.push(1 + axes_b.iter().position(|&x| x == i).unwrap() as isize);
            } else {
                labels_b.push(-(1 + ndim_a as isize + i as isize));
            }
        }

        crate::ncon::ncon_tensors(a, b, &labels_a, &labels_b)
            .expect("tensordot: internal ncon failed — contract axes should be valid")
    }

    fn outer_product(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor {
        let a_shape = a.shape();
        let b_shape = b.shape();
        let mut new_shape = a_shape.to_vec();
        new_shape.extend_from_slice(b_shape);

        let a_br_shape: Vec<usize> = a_shape.iter().copied()
            .chain(std::iter::repeat_n(1, b_shape.len()))
            .collect();
        let b_br_shape: Vec<usize> = std::iter::repeat_n(1, a_shape.len())
            .chain(b_shape.iter().copied())
            .collect();

        let a_br = a.clone().into_shape_with_order(IxDyn(&a_br_shape)).unwrap();
        let b_br = b.clone().into_shape_with_order(IxDyn(&b_br_shape)).unwrap();
        (a_br * b_br).into_shape_with_order(IxDyn(&new_shape)).unwrap()
    }

    fn trace(tensor: &Self::Tensor, axis1: usize, axis2: usize) -> Self::Tensor {
        let ndim = tensor.ndim();
        assert!(axis1 < ndim && axis2 < ndim && axis1 != axis2);

        let other_axes: Vec<usize> = (0..ndim).filter(|&i| i != axis1 && i != axis2).collect();
        let perm: Vec<usize> = other_axes.iter().copied()
            .chain([axis1, axis2].iter().copied())
            .collect();

        let t = Self::transpose(tensor, &perm);
        let shape = t.shape();
        let rest_dim: usize = shape[..ndim - 2].iter().product();
        let trace_dim = shape[ndim - 2];
        let new_shape: Vec<usize> = shape[..ndim - 2].to_vec();

        let t_flat = t.clone().into_shape_with_order(
            IxDyn(&[rest_dim, trace_dim, trace_dim]),
        ).unwrap();

        if new_shape.is_empty() {
            let mut diag_sum = 0.0;
            for k in 0..trace_dim {
                diag_sum += t_flat[[0, k, k]];
            }
            return ndarray::arr0(diag_sum).into_dyn();
        }

        let mut result = ArrayD::zeros(IxDyn(&new_shape));
        for r in 0..rest_dim {
            let mut diag_sum = 0.0;
            for k in 0..trace_dim {
                diag_sum += t_flat[[r, k, k]];
            }
            let mut indices = vec![0usize; new_shape.len()];
            let mut rem = r;
            for (i, &s) in new_shape.iter().enumerate().rev() {
                if s > 0 {
                    indices[i] = rem % s;
                    rem /= s;
                }
            }
            result[IxDyn(&indices)] = diag_sum;
        }
        result
    }

    fn multiply(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor { a * b }

    fn conj(tensor: &Self::Tensor) -> Self::Tensor { tensor.clone() }

    fn convert_to_tensor(data: &[f64], shape: &[usize]) -> Self::Tensor {
        ArrayD::from_shape_vec(IxDyn(shape), data.to_vec()).unwrap()
    }

    fn zeros(shape: &[usize]) -> Self::Tensor { ArrayD::zeros(IxDyn(shape)) }
}
