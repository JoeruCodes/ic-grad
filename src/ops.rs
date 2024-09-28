#[derive(Debug)]
pub enum UnaryOps{
    NOOP,
    EXP2,
    LOG2,
    CAST,
    SIN,
    SQRT,
    RECIP,
    NEG
}
#[derive(Debug)]
pub enum BinaryOps{
    ADD,
    SUB,
    MUL,
    DIV,
    MAX,
    MOD,
    CMPLT
}
#[derive(Debug)]
pub enum ReduceOps{
    SUM,
    MAX
}
#[derive(Debug)]
pub enum TernaryOps{
    MULACC,
    WHERE
}
#[derive(Debug)]
pub enum MovementOps{
    RESHAPE,
    PERMUTE,
    EXPAND,
    PAD,
    SHRINK,
    STRIDE
}
#[derive(Debug)]
pub enum LoadOps{
    EMPTY,
    RAND,
    CONST,
    FROM,
    CONTIGUOUS,
    CUSTOM
}
#[derive(Debug)]
pub enum Ops{
    LoadOps(LoadOps),
    MovementOps(MovementOps),
    TernaryOps(TernaryOps),
    ReduceOps(ReduceOps),
    UnaryOps(UnaryOps),
    BinaryOps(BinaryOps)
}
pub struct Device<'a>{
    _buffers: &'a [&'a str]
}

impl<'a> Device<'a>{
    pub fn new() -> Device<'a>{
        Device { _buffers: &["CPU"] }
    }

    pub const DEFAULT: &'a str = "CPU";

    pub fn canonicalize(device: Option<&'a str>) -> &'a str{return "CPU"}
}