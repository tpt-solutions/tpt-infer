//! Element-wise add requires identical shapes: [1, 784] + [1, 128].
use tpt_infer_graph::{GraphBuilder, Sh};

fn main() {
    let mut b = GraphBuilder::<Sh<1, 784>>::input("x");
    let y = b.add_input::<Sh<1, 128>>("y");
    let _ = b.add(y);
}
