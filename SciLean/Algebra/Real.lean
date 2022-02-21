import SciLean.Mathlib.Algebra.Field.Basic

def ℝ := Float
-- abbrev ℝ := ℝ

def Float.toReal (x : Float) : ℝ := x

namespace Math

  def sqrt : ℝ → ℝ := Float.sqrt
  def pow : ℝ → ℝ → ℝ := Float.pow

  def sin : ℝ → ℝ := Float.sin
  def cos : ℝ → ℝ := Float.cos
  def tan : ℝ → ℝ := Float.tan
  def atan : ℝ → ℝ := Float.atan
  def atan2 : ℝ → ℝ → ℝ := Float.atan2

  def exp : ℝ → ℝ := Float.exp
  def exp2 : ℝ → ℝ := Float.exp2
  def log : ℝ → ℝ := Float.log
  def log2 : ℝ → ℝ := Float.log2
  def log10 : ℝ → ℝ := Float.log10

end Math

namespace ℝ

  def toFloat (x : ℝ) : Float := x
  instance : ToString ℝ := ⟨λ x => x.toFloat.toString⟩
  
  instance : LT ℝ := ⟨λ x y => x.toFloat < y.toFloat⟩
  instance : LE ℝ := ⟨λ x y => x.toFloat ≤ y.toFloat⟩
  instance : OfScientific ℝ := instOfScientificFloat

  instance (x y : ℝ) : Decidable (x < y) := by simp[ℝ] infer_instance done
  -- this kind of breaks with NaNs but I want to make sure that we never get them as division by zero is zero
  instance (x y : ℝ) : Decidable (x = y) := if (x < y) ∨ (y < x) then isFalse (sorry : x≠y) else isTrue (sorry : x=y)
  
  instance : Add ℝ := ⟨λ x y => x.toFloat + y.toFloat⟩
  instance : Sub ℝ := ⟨λ x y => x.toFloat - y.toFloat⟩
  instance : Mul ℝ := ⟨λ x y => x.toFloat * y.toFloat⟩
  instance : Div ℝ := ⟨λ x y => if y = 0.0 then 0.0 else x.toFloat / y.toFloat⟩
  instance : Neg ℝ := ⟨λ x => (-x : Float)⟩

  -- instance : Zero ℝ := ⟨Float.ofNat 0⟩  
  -- instance : One ℝ  := ⟨Float.ofNat 1⟩
  -- instance : OfNat ℝ n := ⟨Float.ofNat n⟩
  -- instance : OfScientific ℝ := ⟨instOfScientificFloat.1⟩

  -- This should override 2.0 interperting as a Float
  @[defaultInstance mid+1]
  instance (priority := high) : OfScientific ℝ := ⟨instOfScientificFloat.1⟩

  -- def natPow (r : ℝ) : Nat → ℝ
  -- | 0 => 1
  -- | n+1 => r * natPow r n

  -- instance : Pow ℝ Nat := ⟨natPow⟩
  instance : HPow ℝ ℝ ℝ := ⟨Math.pow⟩

  -- instance : Numeric ℝ := ⟨λ n => n.toFloat⟩
  instance (n : Nat) : OfNat ℝ n := ⟨n.toFloat⟩
  instance : Coe ℕ ℝ := ⟨λ n => n.toFloat.toReal⟩

  instance : Inv ℝ := ⟨λ x => 1/x⟩

  instance : HPow ℝ ℤ ℝ := 
  ⟨λ x n => 
    match n with
    | Int.ofNat   k => x^(k : ℝ)
    | Int.negSucc k => x^(-(k+1) : ℝ)⟩

--   instance (n : Nat) : OfNat ℝ n := ⟨n.toFloat⟩

-- class Numeric (α : Type u) where
--   ofNat : Nat → α


--   instance : Ring ℝ := 
-- {
-- }

  instance : AddSemigroup ℝ :=
  {
    add_assoc := sorry
  }

  instance : AddCommSemigroup ℝ :=
  {
    add_comm := sorry
  }

  instance : Semigroup ℝ :=
  {
    mul_assoc := sorry
  }

  instance : Semiring ℝ := 
  {
    add_zero := sorry
    zero_add := sorry
    nsmul_zero' := sorry
    nsmul_succ' := sorry
    zero_mul := sorry
    mul_zero := sorry
    one_mul := sorry
    mul_one := sorry
    npow_zero' := sorry
    npow_succ' := sorry

    add_comm := sorry
    left_distrib := sorry
    right_distrib := sorry

    mul_assoc := sorry

    -- mul_add := sorry
    -- add_mul := sorry
    -- ofNat_succ := sorry
  }

  instance : Ring ℝ :=
  {
    sub_eq_add_neg := sorry
    gsmul_zero' := sorry
    gsmul_succ' := sorry
    gsmul_neg' := sorry
    add_left_neg := sorry
  }

  instance : CommRing ℝ := 
  {
    mul_comm := sorry
  }

  instance : Nontrivial ℝ :=
  {
    exists_pair_ne := sorry
  }
  

  instance : Field ℝ := 
  {
    div_eq_mul_inv := sorry 
    mul_inv_cancel := sorry
    inv_zero := sorry
    hpow_succ := sorry
    hpow_neg := sorry
  }
  --   -- by admit
  --   mul_assoc := sorry
  --   add_zero := sorry
  --   zero_add := sorry
  --   add_assoc := sorry
  --   add_comm := sorry
  --   nsmul_zero' := sorry
  --   nsmul_succ' := sorry
  --   zero_mul := sorry
  --   mul_zero := sorry
  --   one_mul := sorry
  --   mul_one := sorry
  --   npow_zero' := sorry
  --   npow_succ' := sorry
  --   mul_add := sorry
  --   add_mul := sorry
  --   ofNat_succ := sorry
  --   sub_eq_add_neg := sorry
  --   gsmul_zero' := sorry
  --   gsmul_succ' := sorry
  --   gsmul_neg' := sorry
  --   add_left_neg := sorry
  --   mul_comm := sorry
  --   exists_pair_ne := sorry
  --   div_eq_mul_inv := sorry
  --   mul_inv_cancel := sorry
  --   inv_zero := sorry
  --   hpow_succ := sorry
  --   hpow_neg := sorry
  -- }

end ℝ



