import SciLean.Categories.Smooth.Operations

namespace SciLean.Smooth

variable {α β γ : Type} 
variable {X Y Z W : Type} [Vec X] [Vec Y] [Vec Z] [Vec W]

def Hom (X Y : Type) [Vec X] [Vec Y] := { f : X → Y // IsSmooth f}

infixr:25 " ⟿ " => Hom

instance {X Y} [Vec X] [Vec Y] : CoeFun (X ⟿ Y) (λ _ => X → Y) := ⟨λ f => f.1⟩
instance {X Y} [Vec X] [Vec Y] (f : X ⟿ Y) : IsSmooth (f : X → Y) := by apply f.2

namespace Hom

  variable (f : X → Y) [IsSmooth f]
  variable (g : X → Y) [IsSmooth g]
  variable (r : ℝ)
 
  instance : IsSmooth (f + g) :=
  by 
    conv => 
      enter [1,x]
      simp
    infer_instance

  instance : IsSmooth (f - g) :=
  by 
    conv => 
      enter [1,x]
      simp
    infer_instance

  instance : IsSmooth (r*f) :=
  by 
    conv => 
      enter [1,x]
      simp
    infer_instance

  instance : IsSmooth (-f) :=
  by
    conv =>
      enter [1,x]
      simp[Neg.neg]
    infer_instance

  instance : Zero (X ⟿ Y) := ⟨⟨0, by infer_instance⟩⟩
  instance : Add (X ⟿ Y) := ⟨λ f g => ⟨f.1 + g.1, by infer_instance⟩⟩
  instance : Sub (X ⟿ Y) := ⟨λ f g => ⟨f.1 - g.1, by infer_instance⟩⟩
  instance : HMul ℝ (X ⟿ Y) (X ⟿ Y) := ⟨λ r f => ⟨r * f.1, by infer_instance⟩⟩
  instance : Neg (X ⟿ Y) := ⟨λ f => ⟨-f.1, by infer_instance⟩⟩

  instance : AddSemigroup (X ⟿ Y) := AddSemigroup.mk sorry
  instance : AddMonoid (X ⟿ Y)    := AddMonoid.mk sorry sorry nsmul_rec sorry sorry
  instance : SubNegMonoid (X ⟿ Y) := SubNegMonoid.mk sorry gsmul_rec sorry sorry sorry
  instance : AddGroup (X ⟿ Y)     := AddGroup.mk sorry
  instance : AddCommGroup (X ⟿ Y) := AddCommGroup.mk sorry

  instance : MulAction ℝ (X ⟿ Y) := MulAction.mk sorry sorry
  instance : DistribMulAction ℝ (X ⟿ Y) := DistribMulAction.mk sorry sorry
  instance : Module ℝ (X ⟿ Y) := Module.mk sorry sorry

  instance : Vec (X ⟿ Y) := Vec.mk


  -- instance {X} [Vec X] [Trait X] : Trait (ℝ ⟿ X) :=
  -- {
  --   sig := ⟨(ℝ × ℝ) → (Trait.sig X).R, 
  --           (ℝ × ℝ) × (Trait.sig X).D,
  --           λ f (I, d) =>  (Trait.sig X).eval (f I) d⟩
  -- }

  open SemiInner 

  instance {X S} [Vec X] [SemiInner' X S] [Vec S.R]
    : SemiInner' (ℝ ⟿ X) S.addInterval :=
  {
    semiInner := λ f g (a,b) => 
      Mathlib.Convenient.integrate a b (λ t => ⟪S| f t, g t⟫) sorry
    testFunction := sorry -- TODO: define test functions on an interval - Probably functions with compact support strictly inside of (a,b). Alternatively, all defivatives vanish at a and b
  }
  open SemiInner in
  @[reducible] instance {X} [Trait X] [Vec X] : Trait (ℝ ⟿ X) := 
    ⟨(Trait.sig X).addInterval⟩

  open SemiInner in
  example {X} [Trait X] [Vec X] [SemiInner X] [Vec (Trait.sig X).R] : SemiInner (ℝ ⟿ X) := SemiInner.mk

  abbrev mk {X Y : Type} [Vec X] [Vec Y] (f : X → Y) [IsSmooth f] : X ⟿ Y := ⟨f, by infer_instance⟩

  -- Right now, I prefer this notation
  macro "fun" xs:Lean.explicitBinders " ⟿ " b:term : term => Lean.expandExplicitBinders `SciLean.Smooth.Hom.mk  xs b
  macro "λ"   xs:Lean.explicitBinders " ⟿ " b:term : term => Lean.expandExplicitBinders `SciLean.Smooth.Hom.mk  xs b

  -- alternative notation
  -- I will decide on one after some use
  macro "funₛ" xs:Lean.explicitBinders " => " b:term : term => Lean.expandExplicitBinders `SciLean.Smooth.Hom.mk  xs b
  macro "λₛ"   xs:Lean.explicitBinders " => " b:term : term => Lean.expandExplicitBinders `SciLean.Smooth.Hom.mk  xs b

  -- Any system in this??? We are basically just restricting a function
  -- to a linear subspace. This does not change the fact if it is 
  -- differentiable or linear.
  instance (f : X → (Y → Z)) [IsSmooth f] [∀ x, IsSmooth (f x)] 
    : IsSmooth (λ x => (λ y ⟿ f x y)) := sorry

  instance (f : X → Y → Z → W) 
    [IsSmooth f] [∀ x, IsSmooth (f x)] [∀ x y, IsSmooth (f x y)] 
    : 
      IsSmooth (λ x => (λ y z ⟿ f x y z)) := sorry

  instance (f : X → Y → Z) 
    [IsSmooth f] [∀ x, IsLin (f x)]
    : 
      IsSmooth (λ x => (λ y ⊸ f x y)) := sorry

  instance (f : X → Y → Z) 
    [IsLin f] [∀ x, IsSmooth (f x)]
    : 
      IsLin (λ x => (λ y ⟿ f x y)) := sorry


  instance (f : X → Y → Z → W) 
    [IsSmooth f] [∀ x, IsSmooth (f x)] [∀ x y, IsLin (f x y)]
    : 
      IsSmooth (λ x => (λ y ⟿ λ z ⊸ f x y z)) := sorry

  instance (f : X → Y → Z → W) 
    [IsLin f] [∀ x, IsSmooth (f x)] [∀ x y, IsLin (f x y)]
    : 
      IsLin (λ x => (λ y ⟿ λ z ⊸ f x y z)) := sorry

  -- set_option synthInstance.maxHeartbeats 50000
  example : X ⟿ X := fun (x : X) ⟿ x
  example : X ⟿ ℝ ⟿ X := fun (x : X) (r : ℝ) ⟿ r*x
  example : X ⟿ ℝ ⟿ X := λ (x : X) (r : ℝ) ⟿ r*x
  -- example : X ⟿ ℝ → X := λ (x : X) (r : ℝ) ⟿ r*x
  example : X → ℝ ⟿ X := λ (x : X) (r : ℝ) ⟿ r*x

  variable (x : X)

  #check λ (x : X) => (λ (t : ℝ) ⟿ x)

  -- This instance is probably a bad idea ... but I'm not sure
  -- instance {X Y Y' Z'} [Vec X] [Vec Y] [CoeFun Y (λ _ => Y' → Z')] : CoeFun (X ⟿ Y) (λ _ => X → Y' → Z') := ⟨λ f x => f.1 x⟩
  -- example : X → ℝ → X := λ (x : X) (r : ℝ) ⟿ r*x

end Hom





