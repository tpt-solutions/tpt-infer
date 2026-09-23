//! A well-typed 2-layer MLP must compile and build successfully.
use tpt_infer_graph::{GraphBuilder, Sh};

fn main() {
    let mut b = GraphBuilder::<Sh<1, 784>>::input("x");
    let w1 = b.add_input::<Sh<784, 128>>("w1");
    let mut b = b.matmul(w1).relu();
    let w2 = b.add_input::<Sh<128, 10>>("w2");
    let b = b.matmul(w2).softmax(-1);

    let g = b.into_graph();
    assert_eq!(g.nodes().len(), 7);
    assert_eq!(g.node(2).unwrap().dims(), &[1, 128]);
    assert_eq!(g.node(6).unwrap().dims(), &[1, 10]);
    assert_eq!(g.topological_sort().unwrap().len(), 7);
}
