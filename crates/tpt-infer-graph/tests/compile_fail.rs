//! trybuild suite: shape mismatches in `GraphBuilder` must fail to compile,
//! while a well-typed graph must compile and run.

#[test]
fn compile_fail_suite() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/*.rs");
    t.pass("tests/compile_fail/pass/*.rs");
}
