//! [1, 784] x [10, 10]: contracting dimensions disagree (784 != 10).
use tpt_infer_graph::{GraphBuilder, Sh};

fn main() {
    let mut b = GraphBuilder::<Sh<1, 784>>::input("x");
    let w = b.add_input::<Sh<10, 10>>("w");
    let _ = b.matmul(w);
}
