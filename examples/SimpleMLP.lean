import SciLean

open SciLean Scalar

structure MLP where
  n : Nat
  m : Nat
  p : Nat
  q : Nat
  weights_input_hidden : Float^[m,p]
  weights_hidden_output : Float^[p,q]
  bias_hidden : Float^[1,p]
  bias_output : Float^[1,q]
  hidden_input : Float^[n,p]
  hidden_output : Float^[n,p]
  final_input : Float^[n,q]
  final_output : Float^[n,q]

namespace MLP

def dot {n : Nat} (x y : Float^[n]) : Float := ∑ i, x[i] * y[i]

def mat_mul {n m p : Nat} (w : Float^[n,m]) (x: Float^[m,p]) :=
  ⊞ (i: Fin n) (j: Fin p) => ∑ k, w[i,k] * x[k,j]

def sigmoid (x : Float) : Float := Id.run do
  return 1.0 / (1.0 + exp (-x))

def transpose {p q : Nat} (x : Float^[p,q]) : Float^[q,p] :=
  ⊞ (i : Fin q) (j: Fin p) => x[j,i]

def sum {n m : Nat} (x : Float^[n,m]) : Float^[1,m] :=
  ⊞ (j : Fin 1) (i: Fin m) => ∑ k, x[k,i]

def addBias {n m : Nat} (x : Float^[n,m]) (y : Float^[1,m]) : Float^[n,m] :=
  ⊞ (i : Fin n) (j: Fin m) => x[i,j] + y[i,j]

def element_mul {n m : Nat} (x : Float^[n,m]) (y : Float^[n,m]) : Float^[n,m] :=
  ⊞ (i : Fin n) (j: Fin m) => x[i,j] * y[i,j]

def sub_one {n p : Nat} (x : Float^[n,p]) : Float^[n,p] :=
  ⊞ (i : Fin n) (j: Fin p) => 1 - x[i,j]

def scalar_mul {n m : Nat} (x : Float^[n,m]) (y : Float) : Float^[n,m] :=
  ⊞ (i : Fin n) (j: Fin m) => y * x[i,j]

def mat_sub {n m : Nat} (x : Float^[n,m]) (y : Float^[n,m]) : Float^[n,m] :=
  ⊞ (i : Fin n) (j: Fin m) => x[i,j] - y[i,j]

--SciLean Function
def softMax {I : Type} [IndexType I]
  (x : Float^[I]) : Float^[I] := Id.run do
  let m := x.reduce (max · ·)
  let x := x.mapMono fun x => x-m
  let x := x.mapMono fun x => exp x
  let w := x.reduce (·+·)
  let x := x.mapMono fun x => x/w
  return x

def forward (mlp : MLP) (X : Float^[mlp.n,mlp.m]) : MLP :=
  let x := addBias (mat_mul X mlp.weights_input_hidden) mlp.bias_hidden
  let y := x.mapMono fun x => sigmoid x
  let j := addBias (mat_mul y mlp.weights_hidden_output) mlp.bias_output
  let k := softMax j
  {mlp with hidden_input := x, hidden_output := y, final_input := j, final_output := k}

def backward (mlp : MLP) (X : Float^[mlp.n,mlp.m]) (y output : Float^[mlp.n,mlp.q]) (learning_rate : Float) : MLP :=
  let output_error := output - y
  let weights_hidden_output_T := transpose mlp.weights_hidden_output
  let hidden_error := element_mul (mat_mul output_error weights_hidden_output_T) (element_mul mlp.hidden_output (sub_one mlp.hidden_output))

  let a := mat_sub mlp.weights_hidden_output (scalar_mul (mat_mul (transpose mlp.hidden_output) output_error) learning_rate)
  let b := mat_sub mlp.bias_output (scalar_mul (sum output_error) learning_rate)
  let c := mat_sub mlp.weights_input_hidden (scalar_mul (mat_mul (transpose X) hidden_error) learning_rate)
  let d := mat_sub mlp.bias_hidden (scalar_mul (sum hidden_error) learning_rate)
  {mlp with weights_hidden_output := a, bias_output := b, weights_input_hidden := c, bias_hidden := d}


end MLP

def main : IO Unit :=

  let batch_size := 2
  let input_features := 3
  let hidden_nodes := 2
  let output_nodes := 2
  let weights_input_hidden := ⊞[0.1, 0.2, 0.3, 0.4, 0.5, 0.6].reshape (Fin input_features × Fin hidden_nodes) (by decide)
  let weights_hidden_output := ⊞[0.7, 0.8, 0.9, 1.0].reshape (Fin hidden_nodes × Fin output_nodes) (by decide)
  let bias_hidden := ⊞[0.1, 0.2].reshape (Fin 1 × Fin hidden_nodes) (by decide)
  let bias_output := ⊞[0.1, 0.2].reshape (Fin 1 × Fin output_nodes) (by decide)
  let hidden_input := ⊞[0.0, 0.0, 0.0, 0.0].reshape (Fin batch_size × Fin hidden_nodes) (by decide)
  let hidden_output := ⊞[0.0, 0.0, 0.0, 0.0].reshape (Fin batch_size × Fin hidden_nodes) (by decide)
  let final_input := ⊞[0.0, 0.0, 0.0, 0.0].reshape (Fin batch_size × Fin output_nodes) (by decide)
  let final_output := ⊞[0.0, 0.0, 0.0, 0.0].reshape (Fin batch_size × Fin output_nodes) (by decide)
  let X := ⊞[1.0, 2.0, 3.0, 4.0, 5.0, 6.0].reshape (Fin batch_size × Fin input_features) (by decide)
  let y := ⊞[1.0, 0.0, 0.0, 1.0].reshape (Fin batch_size × Fin output_nodes) (by decide)
  let output := ⊞[0.0, 0.0, 0.0, 0.0].reshape (Fin batch_size × Fin output_nodes) (by decide)

  let mlp := MLP.mk batch_size input_features hidden_nodes output_nodes weights_input_hidden weights_hidden_output bias_hidden bias_output hidden_input hidden_output final_input final_output
  let mlp := MLP.forward mlp X
  let mlp := MLP.backward mlp X y output 0.001
  let mlp := MLP.forward mlp X
  let mlp := MLP.backward mlp X y output 0.001

  IO.println mlp.final_output

#eval main
