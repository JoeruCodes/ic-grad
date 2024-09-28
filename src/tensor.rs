use lazy_static::lazy_static;
use ndarray::{Array, Data, Dim, IntoDimension, IxDyn, IxDynImpl, OwnedRepr, RawDataClone};
use num_traits::{Float, NumCast, Zero};
use rand::distributions::uniform::SampleUniform;
use std::collections::HashSet;
use std::env;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::{clone::Clone, sync::Arc};
use std::cmp::PartialEq;

use crate::realize::run_schedule;
use crate::{
    helpers::dtypes::{self, DType, FLOAT32}, 
    lazy::LazyBuffer, 
    ops::{Device, LoadOps}
};


pub struct Tensor<'a, E>
where
    E: Clone + 'static + Zero + Float,
{
    lazydata: Option<LazyBuffer<'a, E, OwnedRepr<E>>>, // Placeholder for the actual data
    requires_grad: Option<bool>,            // Whether gradients are required for this tensor
    grad: Option<Arc<Tensor<'a, E>>>,    // The gradient data (can be None if not applicable)
    ctx: Option<String>,                    // Placeholder for the computation context (can be None)
    training: bool,                         // Whether the tensor is in training mode
    no_grad: bool,                          // Disable gradient calculations
}

lazy_static! {
    pub static ref DEFAULT_TYPE: Arc<DType<'static>> = FLOAT32.clone();
}

impl<'a, E> Tensor<'a, E>
where
    E: Clone + Float + Zero + 'static + Default + SampleUniform,
    // S: Data<Elem = E> + RawDataClone,
{
    // Constructor when data is a LazyBuffer
    pub fn new_from_lazybuffer(
        data: LazyBuffer<'a, E, OwnedRepr<E>>,
        device: Option<&str>,
        dtype: Option<Arc<DType<'a>>>,
        requires_grad: Option<bool>,
    ) -> Self {
        let device = Device::canonicalize(device);
        assert!(
            dtype.is_none() || dtype == data.dtype,
            "dtype doesn't match, and casting isn't supported"
        );

        let lazydata = if LazyBuffer::<'a, E, OwnedRepr<E>>::DEVICE == device {
            Some(data)
        } else {
            Some(data.copy_to_device(device))
        };

        Tensor {
            lazydata,
            requires_grad,
            grad: None,
            ctx: None,
            training: false,
            no_grad: false,
        }
    }

    // Constructor when data is a number (int, float)
    pub fn new_from_num(
        data: E,
        requires_grad: Option<bool>,
    ) -> Self {
        Tensor {
            lazydata: Some(LazyBuffer::<'a, E, OwnedRepr<E>>::loadop(
                LoadOps::CONST,
                IxDyn(&[]),
                Some(data),
            )),
            requires_grad,
            grad: None,
            ctx: None,
            training: false,
            no_grad: false,
        }
    }

    // Constructor when data is None or a list
    pub fn new_from_list(
        data: Option<Vec<E>>,
        dtype: Option<Arc<DType<'a>>>,
        requires_grad: Option<bool>,
    ) -> Self {
        let dtype = dtype.unwrap_or_else(|| DEFAULT_TYPE.clone());

        let array_data = match data {
            Some(vec) => Array::from_vec(vec).into_dyn(),
            None => Array::zeros(IxDyn(&[0])),
        };

        let lazydata = LazyBuffer::<'a, E, OwnedRepr<E>>::new_with_dtype(array_data, Some(dtype));

        Tensor {
            lazydata: Some(lazydata),
            requires_grad,
            grad: None,
            ctx: None,
            training: false,
            no_grad: false,
        }
    }

    // Constructor when data is bytes
    pub fn new_from_bytes(
        data: &[u8],
        dtype: Option<Arc<DType<'a>>>,
        requires_grad: Option<bool>,
    ) -> Self
    where
        E: From<u8>,
    {
        let dtype = dtype.unwrap_or_else(|| DType::from_ndarray::<u8>().unwrap().clone());
        let array_data = Array::from_iter(data.iter().cloned().map(<E as From<u8>>::from)).into_dyn();

        let lazydata = LazyBuffer::<'a, E, OwnedRepr<E>>::new_with_dtype(array_data, Some(dtype));

        Tensor {
            lazydata: Some(lazydata),
            requires_grad,
            grad: None,
            ctx: None,
            training: false,
            no_grad: false,
        }
    }

    // Constructor when data is an ndarray
    pub fn new_from_ndarray(
        ndarray_data: Array<E, IxDyn>,
        dtype: Option<Arc<DType<'static>>>,
        requires_grad: Option<bool>,
    ) -> Self {
        let dtype = dtype.unwrap_or_else(|| (*DEFAULT_TYPE).clone());

        let lazydata = if ndarray_data.shape().is_empty() {
            // If ndarray has shape (), treat it as a scalar
            LazyBuffer::<'a, E, OwnedRepr<E>>::loadop(
                LoadOps::CONST,
                IxDyn(&[]),
                Some(ndarray_data[[]]),
            )
        } else {
            // Otherwise, create from CPU data
            LazyBuffer::new_with_dtype(ndarray_data, Some(dtype))
        };

        Tensor {
            lazydata: Some(lazydata),
            requires_grad,
            grad: None,
            ctx: None,
            training: false,
            no_grad: false,
        }
    }

    pub fn device(&self) -> &str{
        LazyBuffer::<'a, E, OwnedRepr<E>>::DEVICE
    }

    pub fn shape(&self) -> Option<Dim<IxDynImpl>>{
        self.lazydata.clone().map(|l|  l.shape)
    }

    pub fn dtype(&self) -> Option<Arc<DType<'a>>>{
        if let Some(l) = &self.lazydata{
            l.dtype.clone()
        }else{
            None
        }
    }

    //pointless
    pub fn corealize(lst: Vec<Tensor<'a, E>>){
        // let seen: HashSet<String> = HashSet::new();

        // let mut sched = Vec::new();

        // for t in lst{
        //     sched.extend(t.lazydata.unwrap().schedule(Some(seen)));
        // }
        // run_schedule(sched);
    }

    //pointless
    pub fn realize(self) -> Tensor<'a, E>{
        run_schedule(self.lazydata.clone().unwrap().schedule::<E>(None));
        return self
    }
    
    pub fn assign(&mut self, x: &mut Tensor<'a, E>) -> &mut Self {
        // Handle the DISK device case
        if self.device().starts_with("DISK") {
            // self.contiguous().realize().lazydata.realized._copyin(x.numpy())
            //return self
            todo!()
        }

        // Ensure that the shapes and devices match
        assert_eq!(
            self.shape(),
            x.shape(),
            "assign shape mismatch {:?} != {:?}",
            self.shape(),
            x.shape()
        );
        assert_eq!(
            self.device(),
            x.device(),
            "device mismatch {} != {}",
            self.device(),
            x.device()
        );

        // Ensure that x.requires_grad is false
        assert!(
            !x.requires_grad.unwrap_or(false),
            "x.requires_grad must be false"
        );

        if self.dtype() == x.dtype() && self.lazydata.as_ref().map(|l| l.realized.clone()).is_some() && env::var("DISALLOW_ASSIGN").is_err(){
            if let Some(l) = &mut x.lazydata{
                l.output_buffer = Some(l.realized.clone());
            }
        }
        self.lazydata = x.lazydata.clone();
        return self;
    }

    pub fn detach(self) -> Tensor<'a, E>{
        return Tensor::new_from_lazybuffer(self.lazydata.clone().unwrap(), Some(self.device()), None, Some(false))
    }

    pub fn ndarray(self){
        assert!(self.dtype().unwrap().np.is_some());
        return self.detach().cast(DType::<'a>::from_ndarray())
    }
}

impl<'a, E> Display for Tensor<'a, E>
    where E: Float + Debug  +SampleUniform + Default
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("<Tensor {:?} on {} with grad {:?}", self.lazydata, self.device(), {
            self.grad.clone().map(|g| g.lazydata.clone())
        }))
    }
}

impl<'a, E> Hash for Tensor<'a, E>
where E: Float + Debug
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let address: *const Self = self as *const Self;
        state.write(format!("{:p}", address).as_bytes());
    }
}
struct Train<'a, E>
    where E: Float + 'static
{
    val: bool,
    prev: Option<bool>,
    tensor_pair: &'a mut Tensor<'a, E>,
}

impl<'a, E> Train<'a, E>
where 
     E: Float
{
    pub fn new(val: Option<bool>, tensor: &'a mut Tensor<'a, E>) -> Self {
        Train { val: val.unwrap_or(true), prev: None, tensor_pair: tensor }
    }

    pub fn init(&mut self) {
        self.prev = Some(self.tensor_pair.training);
        self.tensor_pair.training = self.val;
    }

    pub fn new_init(val: Option<bool>, tensor: &'a mut Tensor<'a, E>) -> Self {
        let mut ret = Train::new(val, tensor);
        ret.init();
        ret
    }

    pub fn drop(&mut self) -> Result<(), String> {
        if let None = self.prev {
            return Err("Train was not initialized".to_string());
        }

        if let Some(p) = self.prev {
            self.tensor_pair.training = p;
        }
        Ok(())
    }
}

impl<'a, E> Drop for Train<'a, E>
where
     E: Float
{
    fn drop(&mut self) {
        if let Some(p) = self.prev {
            self.tensor_pair.training = p;
        } else {
            panic!("Train was not initialized");
        }
    }
}
