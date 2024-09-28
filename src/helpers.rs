use std::{any::TypeId, collections::HashSet, env::var, fmt::Display, hash::Hash};

use lazy_static::lazy_static;
use num_traits::{Num, One, PrimInt, Zero};

#[cfg(target_os = "macos")]
pub const OSX: bool = true;

#[cfg(not(target_os = "macos"))]
pub const OSX: bool = false;


fn dedup<T>(x: Vec<T>) -> Vec<T>
    where T: Eq + PartialEq + Hash
{
    x.into_iter().collect::<HashSet<T>>().into_iter().collect()
} 

#[macro_export]
macro_rules! argfix {
    () => { 
        ()
    };

    ($x: expr) => {
        $x
    };

    ($($x: expr), *) => {
        ($($x), *)
    }
}

fn make_pair<T>(x: T, cnts: Option<usize>) -> Vec<T>
    where T: Clone
{vec![x; cnts.unwrap_or(2)]}

fn argsort<T: PartialOrd>(x: &[T]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..x.len()).collect();
    indices.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());
    indices
}

fn round_up<T>(num: T, amt: T) -> T
where
    T: PrimInt + Zero + One,
{
    if amt.is_zero() {
        panic!("amt cannot be zero");
    }
    ((num + amt - T::one()) / amt) * amt
}

lazy_static!{
    pub static ref DEBUG: bool =  match var("DEBUG").unwrap_or("False".to_string()).to_lowercase().as_str(){
        "true" => true,
        "false" => false,
        _ => false
    };

    pub static ref CI: bool = match var("CI").unwrap_or("False".to_string()).to_lowercase().as_str(){
        "true" => true,
        "false" => false,
        _ => false
    };
}

pub mod dtypes{
    use std::{any::TypeId, fmt::Display, sync::Arc};

    use lazy_static::lazy_static;
    #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    pub struct DType<'a>{
        priority: usize,
        itemsize: usize,
        name: &'a str,
        pub np: Option<TypeId>,
        sz: usize
    }
    
    impl <'a>DType<'a>{
        fn new(priority: usize, itemsize: usize, name: &str, np: Option<TypeId>, sz: Option<usize>) -> DType{
            DType{priority, itemsize, name, np, sz: sz.unwrap_or(1)}
        }
        
        // make this compatible with ndarray
        pub fn is_int(&self) -> bool {
            matches!(
                self.name,
                "char" | "short" | "int" | "long" | "unsigned char" | "unsigned short" | "unsigned int" | "unsigned long"
            )
        }
    
        pub fn is_float(&self) -> bool {
            matches!(self.name, "half" | "float" | "double" | "__bf16")
        }
    
        pub fn is_unsigned(&self) -> bool {
            matches!(
                self.name,
                "unsigned char" | "unsigned short" | "unsigned int" | "unsigned long"
         
            )
        }

        pub fn from_ndarray<T: 'static>() -> Option<Arc<DType<'static>>> {
            let type_id = TypeId::of::<T>();
            if type_id == TypeId::of::<bool>() {
                Some(BOOL.clone())
            } else if type_id == TypeId::of::<f32>() {
                Some(FLOAT32.clone())
            } else if type_id == TypeId::of::<f64>() {
                Some(FLOAT64.clone())
            } else if type_id == TypeId::of::<i8>() {
                Some(INT8.clone())
            } else if type_id == TypeId::of::<i16>() {
                Some(INT16.clone())
            } else if type_id == TypeId::of::<i32>() {
                Some(INT32.clone())
            } else if type_id == TypeId::of::<i64>() {
                Some(INT64.clone())
            } else if type_id == TypeId::of::<u8>() {
                Some(UINT8.clone())
            } else if type_id == TypeId::of::<u16>() {
                Some(UINT16.clone())
            } else if type_id == TypeId::of::<u32>() {
                Some(UINT32.clone())
            } else if type_id == TypeId::of::<u64>() {
                Some(UINT64.clone())
            } else {
                None
            }
        }
    } 
    
    impl<'a> Display for DType<'a>{
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(&format!("dtypes.{}", self.name))
        }
    }
    
    lazy_static!{
        pub static ref BOOL: Arc<DType<'static>> = Arc::new(DType {
            priority: 0,
            itemsize: 1,
            name: "bool",
            np: Some(TypeId::of::<bool>()),
            sz: 1,
        });
    
        // pub static ref FLOAT16: DType<'a> = DType {
        //     priority: 9,
        //     itemsize: 2,
        //     name: "half",
        //     np: Some(TypeId::of::<f16>()),
        //     sz: 1,
        // };
        // pub static ref HALF: DType<'static> = *FLOAT16;
    
        pub static ref FLOAT32: Arc<DType<'static>> = Arc::new(DType {
            priority: 10,
            itemsize: 4,
            name: "float",
            np: Some(TypeId::of::<f32>()),
            sz: 1,
        });
        pub static ref FLOAT: Arc<DType<'static>> = (*FLOAT32).clone();
    
        pub static ref FLOAT64: Arc<DType<'static>> = Arc::new(DType {
            priority: 11,
            itemsize: 8,
            name: "double",
            np: Some(TypeId::of::<f64>()),
            sz: 1,
        });
        pub static ref DOUBLE: Arc<DType<'static>> = (*FLOAT64).clone();
    
        pub static ref INT8: Arc<DType<'static>> = Arc::new(DType {
            priority: 1,
            itemsize: 1,
            name: "char",
            np: Some(TypeId::of::<i8>()),
            sz: 1,
        });
    
        pub static ref INT16: Arc<DType<'static>> = Arc::new(DType {
            priority: 3,
            itemsize: 2,
            name: "short",
            np: Some(TypeId::of::<i16>()),
            sz: 1,
        });
    
        pub static ref INT32: Arc<DType<'static>> = Arc::new(DType {
            priority: 5,
            itemsize: 4,
            name: "int",
            np: Some(TypeId::of::<i32>()),
            sz: 1,
        });
    
        pub static ref INT64: Arc<DType<'static>>= Arc::new(DType {
            priority: 7,
            itemsize: 8,
            name: "long",
            np: Some(TypeId::of::<i64>()),
            sz: 1,
        });
    
        pub static ref UINT8: Arc<DType<'static>> = Arc::new(DType {
            priority: 2,
            itemsize: 1,
            name: "unsigned char",
            np: Some(TypeId::of::<u8>()),
            sz: 1,
        });
    
        pub static ref UINT16: Arc<DType<'static>> = Arc::new(DType {
            priority: 4,
            itemsize: 2,
            name: "unsigned short",
            np: Some(TypeId::of::<u16>()),
            sz: 1,
        });
    
        pub static ref UINT32: Arc<DType<'static>> = Arc::new(DType {
            priority: 6,
            itemsize: 4,
            name: "unsigned int",
            np: Some(TypeId::of::<u32>()),
            sz: 1,
        });
    
        pub static ref UINT64: Arc<DType<'static>> = Arc::new(DType {
            priority: 8,
            itemsize: 8,
            name: "unsigned long",
            np: Some(TypeId::of::<u64>()),
            sz: 1,
        });
    
        // NOTE: bfloat16 isn't supported in Rust's standard library
        // We'll set rust_type to None for bfloat16
        pub static ref BFLOAT16: Arc<DType<'static>> = Arc::new(DType {
            priority: 9,
            itemsize: 2,
            name: "__bf16",
            np: None,
            sz: 1,
        });
    }

    pub static PTR_DTYPE: Option<DType> = None;
    pub static IMAGE_DTYPE: Option<DType> = None;
    pub const IMAGE: i32 = 0; // Junk to remove
}

