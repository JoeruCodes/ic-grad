use std::{fmt::Debug, sync::Arc};
use ndarray::{
    Array, ArrayBase, Axis, Data, DataMut, DataOwned, Dimension, IntoDimension, IxDyn, OwnedRepr, RawDataClone, RemoveAxis, Slice, SliceInfoElem, ViewRepr, Zip
};
use ndarray_rand::RandomExt;
use num_traits::{One, Zero, Float};
use rand::distributions::Uniform;
use ndarray::s;
use crate::{helpers::dtypes::DType, ops::{BinaryOps, Device, LoadOps, Ops, ReduceOps, TernaryOps, UnaryOps}};
#[derive(Clone, Debug)]
pub struct RawCpuBuffer<T> {
    x: T,
}

impl<T> RawCpuBuffer<T> {
    pub fn new(x: T) -> RawCpuBuffer<T> {
        RawCpuBuffer { x }
    }

    pub fn to_cpu(self) -> T {
        self.x
    }
}

#[derive(Clone, Debug)]
pub struct LazyBuffer<'a, E, S = OwnedRepr<E>>
where
    S: Data<Elem = E> + RawDataClone,
{
    _np: ArrayBase<S, IxDyn>,
    pub realized: RawCpuBuffer<ArrayBase<S, IxDyn>>,
    pub dtype: Option<Arc<DType<'a>>>,
    pub shape: IxDyn,
    pub output_buffer: Option<RawCpuBuffer<ArrayBase<S, IxDyn>>>
}

impl<'a, E, S> LazyBuffer<'a, E, S>
where
    E: Clone + 'static + Zero + Float,
    S: Data<Elem = E> + RawDataClone,
{
    pub const DEVICE: &'static str = "CPU";
    pub fn new_with_dtype(
        buf: ArrayBase<S, IxDyn>,
        dtype: Option<Arc<DType<'a>>>,
    ) -> LazyBuffer<'a, E, S> {
        LazyBuffer {
            _np: buf.clone(),
            realized: RawCpuBuffer::new(buf.clone()),
            dtype,
            shape: buf.dim().clone(),
            output_buffer: None
        }
    }
    pub fn new(buf: ArrayBase<S, IxDyn>) -> LazyBuffer<'a, E, S> {
        LazyBuffer {
            _np: buf.clone(),
            realized: RawCpuBuffer::new(buf.clone()),
            dtype: DType::from_ndarray::<E>(),
            shape: buf.dim().clone(),
            output_buffer: None
        }
    }

    pub fn base(&self) -> &Self {
        self
    }

    pub fn schedule<T>(&self, _seen: Option<T>) -> Vec<T> {
        Vec::new()
    }

    pub fn is_unrealized_contiguous_const(&self) -> bool {
        false
    }

    pub fn copy_to_device(self, _device: &str) -> Self {
        self
    }

    pub fn from_cpu(x: Array<E, IxDyn>) -> LazyBuffer<'a, E, OwnedRepr<E>> {
        LazyBuffer::<'a, E, OwnedRepr<E>>::new(x)
    }

    pub fn loadop(
        op: LoadOps,
        shape: IxDyn,
        arg: Option<E>,
    ) -> LazyBuffer<'a, E, OwnedRepr<E>>
    where
        E: Clone
            + Default
            + One
            + Zero
            + ndarray_rand::rand_distr::uniform::SampleUniform
            + 'static,
    {
        let shape = shape.into_dimension();
        match op {
            LoadOps::RAND => {
                let arr = Array::<E, IxDyn>::random(shape.clone(), Uniform::new(E::zero(), E::one()));
                LazyBuffer::<'a, E, OwnedRepr<E>>::new(arr)
            }
            LoadOps::CONST => {
                let mut arr = Array::<E, IxDyn>::zeros(shape.clone());
                arr.fill(arg.unwrap_or(E::zero()));
                LazyBuffer::<'a, E, OwnedRepr<E>>::new(arr)
            }
            LoadOps::EMPTY => {
                let arr = Array::<E, IxDyn>::zeros(shape.clone());
                LazyBuffer::<'a, E, OwnedRepr<E>>::new(arr)
            }
            _ => unimplemented!(),
        }
    }

    pub fn contiguous<T>(x: T) -> T {
        x
    }

    pub fn r#const(&self, x: E) -> LazyBuffer<'a, E, OwnedRepr<E>>
    where
        E: Default + Zero,
    {
        let mut ret = Array::<E, IxDyn>::zeros(self.shape.clone());
        ret.fill(x);
        LazyBuffer::<'a, E, OwnedRepr<E>>::new(ret)
    }

    pub fn bitcast(&'a self, _dtype: DType) -> LazyBuffer<'a, E, ViewRepr<&'a E>> {
        LazyBuffer::<'a, E, ViewRepr<&E>>::new(self._np.view())
    }

    pub fn e(
        self,
        op: Ops,
        srcs: Vec<&LazyBuffer<'a, E, S>>,
    ) -> LazyBuffer<'a, E, OwnedRepr<E>>
    where
        E: Clone + 'static + std::ops::Neg<Output = E> + Float + Zero + One,
        S: Data<Elem = E> + RawDataClone,
    {

        let ret_array = match op {
            Ops::UnaryOps(uop) => match uop {
                UnaryOps::NEG => {
                    // Unary negation using mapv
                    self._np.mapv(|x| -x)
                }
                UnaryOps::EXP2 => {
                    self._np.mapv(|x| x.exp2())
                }
                UnaryOps::LOG2 => {
                    self._np.mapv(|x| x.log2())
                }
                UnaryOps::SIN => {
                    self._np.mapv(|x| x.sin())
                }
                UnaryOps::SQRT => {
                    self._np.mapv(|x| x.sqrt())
                }
                _ => unimplemented!("{:?}", uop),
            },
            Ops::BinaryOps(bop) => {
                let other = &srcs[0]._np;
                match bop {
                    BinaryOps::ADD => &self._np + other,
                    BinaryOps::SUB => &self._np - other,
                    BinaryOps::MUL => &self._np * other,
                    BinaryOps::DIV => &self._np / other,
                    BinaryOps::MAX => {
                        // Use Zip and for_each to compute element-wise maximum
                        let mut result = Array::zeros(self._np.raw_dim());
                        Zip::from(&mut result)
                            .and(&self._np)
                            .and(other)
                            .for_each(|r, &x, &y| {
                                *r = x.max(y);
                            });
                        result
                    }
                    BinaryOps::CMPLT => {
                        // Use Zip and for_each for element-wise comparison
                        let mut result = Array::zeros(self._np.raw_dim());
                        Zip::from(&mut result)
                            .and(&self._np)
                            .and(other)
                            .for_each(|r, &x, &y| {
                                *r = if x < y { E::one() } else { E::zero() };
                            });
                        result
                    }
                    _ => unimplemented!("{:?}", bop),
                }
            }
            Ops::TernaryOps(top) => match top {
                TernaryOps::WHERE => {
                    let cond = &self._np;
                    let x = &srcs[0]._np;
                    let y = &srcs[1]._np;
                    let mut result = Array::zeros(cond.raw_dim());
                    Zip::from(&mut result)
                        .and(cond)
                        .and(x)
                        .and(y)
                        .for_each(|r, &c, &x_val, &y_val| {
                            *r = if c != E::zero() { x_val } else { y_val };
                        });
                    result
                }
                _ => unimplemented!("{:?}", top),
            },
            _ => unimplemented!("{:?}", op),
        };

        // Determine the dtype (simplified)
        let dtype = if srcs.is_empty() {
            self.dtype
        } else {
            // Collect all dtypes from srcs
            let mut all_dtypes = srcs
                .into_iter()
                .filter_map(|x| x.dtype.clone())
                .collect::<Vec<Arc<DType>>>();

            // Add self.dtype to the list
            if let Some(self_dtype) = self.dtype {
                all_dtypes.push(self_dtype);
            }

            // Find the maximum dtype
            all_dtypes.into_iter().max()
        };

        // Create the new LazyBuffer with ret_array and dtype
        LazyBuffer::<'a, E, OwnedRepr<E>>::new_with_dtype(ret_array, dtype)
    }
    pub fn r(
        self,
        op: ReduceOps,
        new_shape: IxDyn,
    ) -> Result<LazyBuffer<'a, E, OwnedRepr<E>>, String> {

        if self.shape.ndim() != new_shape.ndim() {
            return Err("Reduce shapes must have same dimensions".to_string());
        }

        // Determine axes to reduce over
        let axes_to_reduce: Vec<usize> = self
            .shape
            .slice()
            .iter()
            .zip(new_shape.slice().iter())
            .enumerate()
            .filter_map(|(i, (&a, &b))| if a != b { Some(i) } else { None })
            .collect();

        // Start with the original array
        let mut result_array = self._np.to_owned();

        for &ax in axes_to_reduce.iter().rev() {
            result_array = match op {
                ReduceOps::SUM => {
                    let reduced = result_array.sum_axis(Axis(ax));
                    reduced.insert_axis(Axis(ax))
                }
                ReduceOps::MAX => {
                    let reduced = result_array.map_axis(Axis(ax), |x| {
                        x.iter().cloned().fold(E::min_value(), E::max)
                    });
                    reduced.insert_axis(Axis(ax))
                }
            };
        }

        Ok(LazyBuffer::new_with_dtype(result_array, self.dtype))
    }

    pub fn reshape(
        self,
        new_shape: IxDyn, // Accept dynamic shape
    ) -> Result<LazyBuffer<'a, E, OwnedRepr<E>>, String> {
        // Try to reshape the array
        let reshaped_np = match self._np.to_shape(new_shape.clone()) {
            Ok(array) => array.to_owned(), // Convert the reshaped array to an owned version
            Err(e) => return Err(format!("Reshape failed: {}", e)),
        };

        // Return a new LazyBuffer with the reshaped array
        Ok(LazyBuffer::<'a, E, OwnedRepr<E>>::new_with_dtype(
            reshaped_np, 
            self.dtype.clone(),
        ))
    }

    pub fn expand(
        self,
        new_shape: IxDyn, // Accept dynamic shape for broadcasting
    ) -> Result<LazyBuffer<'a, E, OwnedRepr<E>>, String> {
        // Try to broadcast the array
        let broadcasted_np = match self._np.broadcast(new_shape) {
            Some(array) => array.to_owned(), // Convert the broadcasted view to an owned array
            None => return Err("Broadcast failed: incompatible shapes".to_string()),
        };

        // Return a new LazyBuffer with the broadcasted array
        Ok(LazyBuffer::<'a, E, OwnedRepr<E>>::new_with_dtype(
            broadcasted_np,
            self.dtype,
        ))
    }

    pub fn shrink(
        self,
        arg: Vec<(usize, usize)>, // Accept a list of (start, end) tuples for slicing
    ) -> Result<LazyBuffer<'a, E, OwnedRepr<E>>, String> {
        // Create a vector of slices based on the input argument
        let slices: Vec<SliceInfoElem> = arg
            .iter()
            .map(|&(start, end)| SliceInfoElem::Slice {
                start: start as isize,
                end: Some(end as isize),
                step: 1,
            })
            .collect();

        // Try to apply the slicing to the array
        let sliced_np = self._np.slice(slices.as_slice()).to_owned();
        // Return a new LazyBuffer with the sliced array
        Ok(LazyBuffer::<'a, E, OwnedRepr<E>>::new_with_dtype(
            sliced_np,
            self.dtype.clone(),
        ))
    }
    pub fn permute(
        self,
        arg: Vec<usize>, // Accept a vector of indices representing the new axis order
    ) -> Result<LazyBuffer<'a, E, OwnedRepr<E>>, String> {
        // Try to permute the axes of the array
        let permuted_np = self._np.clone().permuted_axes(arg);

        // Return a new LazyBuffer with the permuted array
        Ok(LazyBuffer::<'a, E, OwnedRepr<E>>::new_with_dtype(
            permuted_np.to_owned(),
            self.dtype.clone(),
        ))
    }

    pub fn pad(
        self,
        arg: Vec<(usize, usize)>, // Accept a list of (pad_before, pad_after) for each dimension
    ) -> Result<LazyBuffer<'a, E, OwnedRepr<E>>, String> {
        // Get the shape of the original array
        let original_shape = self._np.shape();
        
        // Calculate the new shape after padding
        let new_shape: Vec<usize> = original_shape
            .iter()
            .zip(arg.iter())
            .map(|(&dim_size, &(pad_before, pad_after))| dim_size + pad_before + pad_after)
            .collect();

        // Create a new array with zeros (or any other fill value) of the padded shape
        let mut padded_array = Array::zeros(IxDyn(&new_shape));

        // Determine the slice for the original array within the padded array
        let slices: Vec<_> = arg
            .iter()
            .map(|&(pad_before, _)| {
                ndarray::SliceInfoElem::Slice {
                    start: pad_before as isize,
                    end: None,
                    step: 1,
                }
            })
            .collect();

        // Copy the original array into the correctly padded area of the new array
        padded_array
            .slice_mut(slices.as_slice())
            .assign(&self._np);

        // Return a new LazyBuffer with the padded array
        Ok(LazyBuffer::<'a, E, OwnedRepr<E>>::new_with_dtype(
            padded_array,
            self.dtype,
        ))
    }

    pub fn stride(
        self,
        arg: Vec<usize>, // Accept a list of strides for each dimension
    ) -> Result<LazyBuffer<'a, E, OwnedRepr<E>>, String> {
        // Create a vector of slices where each dimension is sliced with a step/stride
        let slices: Vec<SliceInfoElem> = arg
            .iter()
            .map(|&stride| SliceInfoElem::Slice {
                start: 0,
                end: None,
                step: stride as isize, // Use the provided stride for each dimension
            })
            .collect();

        // Try to apply the slicing with strides to the array
        let strided_np = self._np.slice(slices.as_slice());

        // Return a new LazyBuffer with the strided array
        Ok(LazyBuffer::<'a, E, OwnedRepr<E>>::new_with_dtype(
            strided_np.to_owned(),
            self.dtype,
        ))
    }
}