use crate::circuit::ops::poly::PolyOp;
use crate::circuit::*;
use crate::tensor::{DataFormat, KernelFormat};
use crate::tensor::{Tensor, TensorType, ValTensor, VarTensor};
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    dev::MockProver,
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::bn256::Fr as F;
use halo2curves::ff::{Field, PrimeField};
use itertools::Itertools;
#[cfg(not(any(
    all(target_arch = "wasm32", target_os = "unknown"),
    not(feature = "ezkl")
)))]
use ops::lookup::LookupOp;
use ops::region::RegionCtx;
use rand::rngs::OsRng;
use std::marker::PhantomData;

#[derive(Default)]
struct TestParams;

// NOTE: Split to keep file sizes manageable for patch processing.
// Contents are included verbatim from the two files below.
include!("tests_part1.rs");
include!("tests_part2.rs");

/// Step 8: Probabilistic execution test for MatMul.
/// This is gated behind `feature="ezkl"` since it generates a real proof (not just a MockProver run).
#[cfg(test)]
#[cfg(all(
    feature = "ezkl",
    not(all(target_arch = "wasm32", target_os = "unknown"))
))]
mod probabilistic_matmul {
    use super::*;

    use halo2_proofs::poly::kzg::{
        commitment::KZGCommitmentScheme,
        multiopen::{ProverSHPLONK, VerifierSHPLONK},
        strategy::SingleStrategy,
    };
    use snark_verifier::system::halo2::transcript::evm::EvmTranscript;

    use crate::commands::{ExecutionMode, RunArgs};
    use crate::circuit::ops::probabilistic::FreivaldsCheck;

    const K: usize = 8;
    const LEN: usize = 2;

    #[derive(Clone)]
    struct MatmulCircuit<Fp: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<Fp>; 2],
        _marker: PhantomData<Fp>,
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
                    || "probabilistic_matmul_region",
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

    #[derive(Clone)]
    struct FreivaldsSoundnessCircuit<Fp: PrimeField + TensorType + PartialOrd> {
        a: ValTensor<Fp>,
        b: ValTensor<Fp>,
        c: ValTensor<Fp>,
        _marker: PhantomData<Fp>,
    }

    impl Circuit<F> for FreivaldsSoundnessCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            // Keep the same "BaseConfig + wrap-across-columns" config as the other MatMul test.
            let a = VarTensor::new_advice(cs, K, 1, LEN * LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN * LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN * LEN);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "probabilistic_soundness_region",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);

                        // Deterministic seed so the test is stable.
                        let seed_t = Tensor::from(std::iter::once(Value::known(F::from(0u64))));
                        let seed = ValTensor::from(seed_t);

                        let check = FreivaldsCheck::new(&self.a, &self.b, &self.c);

                        // Access the same einsum chip/config that BaseConfig uses.
                        let einsums = config.einsums();

                        check
                            .layout(einsums, &mut region, &seed, 0, &CheckMode::SAFE)
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();

            Ok(())
        }
    }

    #[test]
    fn test_probabilistic_matmul() {
        // A and B are 2x2 so output is 2x2.
        let mut a = Tensor::from((0..(LEN * LEN)).map(|i| Value::known(F::from((i + 1) as u64))));
        a.reshape(&[LEN, LEN]).unwrap();

        let mut b = Tensor::from((0..(LEN * LEN)).map(|i| Value::known(F::from((i + 1) as u64))));
        b.reshape(&[LEN, LEN]).unwrap();

        let circuit = MatmulCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        // Configure probabilistic execution: only MatMul is probabilistic.
        let mut run_args = RunArgs::default();
        run_args.execution_mode = ExecutionMode::Probabilistic;
        run_args.prob_ops = vec!["MatMul".to_string()];

        let params = crate::pfsys::srs::gen_srs::<KZGCommitmentScheme<_>>(K as u32);

        let pk = crate::pfsys::create_keys::<
            KZGCommitmentScheme<halo2curves::bn256::Bn256>,
            MatmulCircuit<F>,
        >(&circuit, &params, true)
        .unwrap();

        let prover = crate::pfsys::create_proof_circuit::<
            KZGCommitmentScheme<_>,
            _,
            ProverSHPLONK<_>,
            VerifierSHPLONK<_>,
            SingleStrategy<_>,
            _,
            EvmTranscript<_, _, _, _>,
            EvmTranscript<_, _, _, _>,
        >(
            circuit.clone(),
            vec![],
            &params,
            &pk,
            // Safe mode: ensures the generated proof verifies correctly under the selected execution mode.
            CheckMode::SAFE,
            Some(run_args),
            None,
        );

        assert!(prover.is_ok());
    }

    /// Step 9: Soundness test for probabilistic MatMul.
    ///
    /// This uses the same (A,B) as the MatMul test, but provides an intentionally *wrong*
    /// witness for C and asserts that the proof does *not* verify.
    #[test]
    fn test_probabilistic_soundness() {
        // A and B are 2x2 so output is 2x2.
        let mut a = Tensor::from((0..(LEN * LEN)).map(|i| Value::known(F::from((i + 1) as u64))));
        a.reshape(&[LEN, LEN]).unwrap();

        let mut b = Tensor::from((0..(LEN * LEN)).map(|i| Value::known(F::from((i + 1) as u64))));
        b.reshape(&[LEN, LEN]).unwrap();

        // Correct C for A=B=[[1,2],[3,4]] is [[7,10],[15,22]].
        let mut c = Tensor::from(vec![
            Value::known(F::from(7u64)),
            Value::known(F::from(10u64)),
            Value::known(F::from(15u64)),
            Value::known(F::from(22u64)),
        ]);
        c.reshape(&[LEN, LEN]).unwrap();

        // Perturb the witness (C[0,0] += 1).
        c[0] = Value::known(F::from(8u64));

        let circuit = FreivaldsSoundnessCircuit::<F> {
            a: ValTensor::from(a),
            b: ValTensor::from(b),
            c: ValTensor::from(c),
            _marker: PhantomData,
        };

        // Configure probabilistic execution (matches Step 8 setup).
        let mut run_args = RunArgs::default();
        run_args.execution_mode = ExecutionMode::Probabilistic;
        run_args.prob_ops = vec!["MatMul".to_string()];

        let params = crate::pfsys::srs::gen_srs::<KZGCommitmentScheme<_>>(K as u32);

        let pk = crate::pfsys::create_keys::<
            KZGCommitmentScheme<halo2curves::bn256::Bn256>,
            FreivaldsSoundnessCircuit<F>,
        >(&circuit, &params, true)
        .unwrap();

        let prover = crate::pfsys::create_proof_circuit::<
            KZGCommitmentScheme<_>,
            _,
            ProverSHPLONK<_>,
            VerifierSHPLONK<_>,
            SingleStrategy<_>,
            _,
            EvmTranscript<_, _, _, _>,
            EvmTranscript<_, _, _, _>,
        >(
            circuit,
            vec![],
            &params,
            &pk,
            CheckMode::SAFE,
            Some(run_args),
            None,
        );

        assert!(
            prover.is_err(),
            "expected probabilistic MatMul proof to fail verification with a perturbed C witness"
        );
    }
}

