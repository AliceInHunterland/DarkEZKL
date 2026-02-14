#[cfg(test)]
mod matmul {

    use super::*;

    const K: usize = 9;
    const LEN: usize = 3;

    #[derive(Clone)]
    struct MatmulCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MatmulCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN * LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN * LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN * LEN);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Einsum {
                                    equation: "ij,jk->ik".to_string(),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();

            Ok(())
        }
    }

    #[test]
    fn matmulcircuit() {
        // parameters
        let mut a =
            Tensor::from((0..(LEN + 1) * LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        a.reshape(&[LEN, LEN + 1]).unwrap();

        let mut w = Tensor::from((0..LEN + 1).map(|i| Value::known(F::from((i + 1) as u64))));
        w.reshape(&[LEN + 1, 1]).unwrap();

        let circuit = MatmulCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(w)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod matmul_col_overflow_double_col {
    use super::*;

    const K: usize = 5;
    const LEN: usize = 6;
    const NUM_INNER_COLS: usize = 2;

    #[derive(Clone)]
    struct MatmulCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MatmulCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN * LEN * LEN);
            let b = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN * LEN * LEN);
            let output = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN * LEN * LEN);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, NUM_INNER_COLS, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Einsum {
                                    equation: "ij,jk->ik".to_string(),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn matmulcircuit() {
        // parameters
        let mut a = Tensor::from((0..LEN * LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        a.reshape(&[LEN, LEN]).unwrap();

        let mut w = Tensor::from((0..LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        w.reshape(&[LEN, 1]).unwrap();

        let circuit = MatmulCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(w)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod matmul_col_overflow {
    use super::*;

    const K: usize = 5;
    const LEN: usize = 6;

    #[derive(Clone)]
    struct MatmulCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MatmulCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Einsum {
                                    equation: "ij,jk->ik".to_string(),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn matmulcircuit() {
        // parameters
        let mut a = Tensor::from((0..LEN * LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        a.reshape(&[LEN, LEN]).unwrap();

        let mut w = Tensor::from((0..LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        w.reshape(&[LEN, 1]).unwrap();

        let circuit = MatmulCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(w)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
#[cfg(all(
    feature = "ezkl",
    not(all(target_arch = "wasm32", target_os = "unknown"))
))]
mod matmul_col_ultra_overflow_double_col {

    use halo2_proofs::poly::kzg::{
        commitment::KZGCommitmentScheme,
        multiopen::{ProverSHPLONK, VerifierSHPLONK},
        strategy::SingleStrategy,
    };
    use halo2_proofs::transcript::{Blake2bWrite, Blake2bRead, Challenge255};
    use halo2curves::bn256::G1Affine;
    use std::io::Cursor;

    use super::*;

    const K: usize = 4;
    const LEN: usize = 10;
    const NUM_INNER_COLS: usize = 2;

    #[derive(Clone)]
    struct MatmulCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MatmulCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN * LEN * LEN);
            let b = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN * LEN * LEN);
            let output = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN * LEN * LEN);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, NUM_INNER_COLS, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Einsum {
                                    equation: "ij,jk->ik".to_string(),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    #[ignore]
    fn matmulcircuit() {
        // get some logs fam
        crate::logger::init_logger();
        // parameters
        let mut a = Tensor::from((0..LEN * LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        a.reshape(&[LEN, LEN]).unwrap();

        let mut w = Tensor::from((0..LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        w.reshape(&[LEN, 1]).unwrap();

        let circuit = MatmulCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(w)],
            _marker: PhantomData,
        };

        let params = crate::pfsys::srs::gen_srs::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<_>,
        >(K as u32);

        let pk = crate::pfsys::create_keys::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<halo2curves::bn256::Bn256>,
            MatmulCircuit<F>,
        >(&circuit, &params, true)
        .unwrap();

        let prover = crate::pfsys::create_proof_circuit::<
            KZGCommitmentScheme<_>,
            _,
            ProverSHPLONK<_>,
            VerifierSHPLONK<_>,
            SingleStrategy<_>,
            Challenge255<G1Affine>,
            Blake2bWrite<Vec<u8>, G1Affine, Challenge255<G1Affine>>,
            Blake2bRead<Cursor<Vec<u8>>, G1Affine, Challenge255<G1Affine>>,
        >(
            circuit.clone(),
            vec![],
            &params,
            &pk,
            // use safe mode to verify that the proof is correct
            CheckMode::SAFE,
            None,
            None,
        );

        assert!(prover.is_ok());
    }
}

#[cfg(test)]
#[cfg(all(
    feature = "ezkl",
    not(all(target_arch = "wasm32", target_os = "unknown"))
))]
mod matmul_col_ultra_overflow {

    use halo2_proofs::poly::kzg::{
        commitment::KZGCommitmentScheme,
        multiopen::{ProverSHPLONK, VerifierSHPLONK},
        strategy::SingleStrategy,
    };
    use itertools::Itertools;
    use halo2_proofs::transcript::{Blake2bWrite, Blake2bRead, Challenge255};
    use halo2curves::bn256::G1Affine;
    use std::io::Cursor;

    use super::*;

    const K: usize = 4;
    const LEN: usize = 10;

    #[derive(Clone)]
    struct MatmulCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MatmulCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Einsum {
                                    equation: "ij,jk->ik".to_string(),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    #[ignore]
    fn matmulcircuit() {
        // get some logs fam
        crate::logger::init_logger();
        // parameters
        let mut a = Tensor::from((0..LEN * LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        a.reshape(&[LEN, LEN]).unwrap();

        let mut w = Tensor::from((0..LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        w.reshape(&[LEN, 1]).unwrap();

        let circuit = MatmulCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(w)],
            _marker: PhantomData,
        };

        let params = crate::pfsys::srs::gen_srs::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<_>,
        >(K as u32);

        let pk = crate::pfsys::create_keys::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<halo2curves::bn256::Bn256>,
            MatmulCircuit<F>,
        >(&circuit, &params, true)
        .unwrap();

        let prover = crate::pfsys::create_proof_circuit::<
            KZGCommitmentScheme<_>,
            _,
            ProverSHPLONK<_>,
            VerifierSHPLONK<_>,
            SingleStrategy<_>,
            Challenge255<G1Affine>,
            Blake2bWrite<Vec<u8>, G1Affine, Challenge255<G1Affine>>,
            Blake2bRead<Cursor<Vec<u8>>, G1Affine, Challenge255<G1Affine>>
        >(
            circuit.clone(),
            vec![],
            &params,
            &pk,
            // use safe mode to verify that the proof is correct
            CheckMode::SAFE,
            None,
            None,
        );

        assert!(prover.is_ok());
    }
}

#[cfg(test)]
mod dot {
    use ops::poly::PolyOp;

    use super::*;

    const K: usize = 4;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Einsum {
                                    equation: "i,i->".to_string(),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn dotcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let b = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod dot_col_overflow_triple_col {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 50;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            // used for constants in the padding
            let _fixed = cs.fixed_column();
            cs.enable_constant(_fixed);

            let a = VarTensor::new_advice(cs, K, 3, LEN);
            let b = VarTensor::new_advice(cs, K, 3, LEN);
            let output = VarTensor::new_advice(cs, K, 3, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 3, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Einsum {
                                    equation: "i,i->".to_string(),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn dotcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let b = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod dot_col_overflow {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 50;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Einsum {
                                    equation: "i,i->".to_string(),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn dotcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let b = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod sum {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 1],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Sum { axes: vec![0] }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn sumcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod sum_col_overflow_double_col {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 20;
    const NUM_INNER_COLS: usize = 2;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 1],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN);
            let b = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN);
            let output = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, NUM_INNER_COLS, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Sum { axes: vec![0] }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn sumcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod sum_col_overflow {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 20;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 1],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Sum { axes: vec![0] }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn sumcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod composition {

    use super::*;

    const K: usize = 9;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            // lots of stacked dot products
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        let _ = config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Einsum {
                                    equation: "i,i->".to_string(),
                                }),
                            )
                            .unwrap();
                        let _ = config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Einsum {
                                    equation: "i,i->".to_string(),
                                }),
                            )
                            .unwrap();
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Einsum {
                                    equation: "i,i->".to_string(),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn dotcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let b = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod conv {

    use super::*;

    const K: usize = 22;
    const LEN: usize = 100;

    #[derive(Clone)]
    struct ConvCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: Vec<ValTensor<F>>,
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for ConvCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, (LEN + 1) * LEN);
            let b = VarTensor::new_advice(cs, K, 1, (LEN + 1) * LEN);
            let output = VarTensor::new_advice(cs, K, 1, (LEN + 1) * LEN);

            // column for constants
            let _constant = VarTensor::constant_cols(cs, K, 8, false);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Conv {
                                    padding: vec![(1, 1); 2],
                                    stride: vec![2; 2],
                                    group: 1,
                                    data_format: DataFormat::default(),
                                    kernel_format: KernelFormat::default(),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn convcircuit() {
        // parameters
        let kernel_height = 2;
        let kernel_width = 3;
        let image_height = 5;
        let image_width = 7;
        let in_channels = 3;
        let out_channels = 2;

        let mut image =
            Tensor::from((0..in_channels * image_height * image_width).map(|_| F::random(OsRng)));
        image
            .reshape(&[1, in_channels, image_height, image_width])
            .unwrap();
        image.set_visibility(&crate::graph::Visibility::Private);

        let image = ValTensor::try_from(image).unwrap();

        let mut kernels = Tensor::from(
            (0..{ out_channels * in_channels * kernel_height * kernel_width })
                .map(|_| F::random(OsRng)),
        );
        kernels
            .reshape(&[out_channels, in_channels, kernel_height, kernel_width])
            .unwrap();
        kernels.set_visibility(&crate::graph::Visibility::Private);

        let kernels = ValTensor::try_from(kernels).unwrap();
        let mut bias = Tensor::from((0..{ out_channels }).map(|_| F::random(OsRng)));
        bias.set_visibility(&crate::graph::Visibility::Private);

        let bias = ValTensor::try_from(bias).unwrap();

        let circuit = ConvCircuit::<F> {
            inputs: [image, kernels, bias].to_vec(),
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn convcircuitnobias() {
        // parameters
        let kernel_height = 2;
        let kernel_width = 2;
        let image_height = 4;
        let image_width = 5;
        let in_channels = 3;
        let out_channels = 2;

        let mut image =
            Tensor::from((0..in_channels * image_height * image_width).map(|i| F::from(i as u64)));
        image
            .reshape(&[1, in_channels, image_height, image_width])
            .unwrap();
        image.set_visibility(&crate::graph::Visibility::Private);

        let mut kernels = Tensor::from(
            (0..{ out_channels * in_channels * kernel_height * kernel_width })
                .map(|i| F::from(i as u64)),
        );
        kernels
            .reshape(&[out_channels, in_channels, kernel_height, kernel_width])
            .unwrap();
        kernels.set_visibility(&crate::graph::Visibility::Private);

        let image = ValTensor::try_from(image).unwrap();
        let kernels = ValTensor::try_from(kernels).unwrap();

        let circuit = ConvCircuit::<F> {
            inputs: [image, kernels].to_vec(),
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

