use super::errors::GraphError;
use super::utilities::{dequantize, scale_to_multiplier};
use super::vars::VarVisibility;

use crate::fieldutils::{felt_to_f64, IntegerRep};
use crate::pfsys::PrettyElements;
use crate::tensor::Tensor;
use crate::EZKL_BUF_CAPACITY;

use halo2curves::bn256::{Fr as Fp, G1Affine};
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use pyo3::types::PyDict;
#[cfg(feature = "python-bindings")]
use pyo3::types::PyDictMethods;
#[cfg(feature = "python-bindings")]
use pyo3::IntoPyObject;

use serde::{Deserialize, Serialize};

#[cfg(feature = "python-bindings")]
use crate::pfsys::field_to_string;

/// Result from a forward pass
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct GraphWitness {
    /// The inputs of the forward pass
    pub inputs: Vec<Vec<Fp>>,
    /// The prettified outputs of the forward pass, we use a String to maximize compatibility with Python and JS clients
    pub pretty_elements: Option<PrettyElements>,
    /// The output of the forward pass
    pub outputs: Vec<Vec<Fp>>,
    /// Any hashes of inputs generated during the forward pass
    pub processed_inputs: Option<crate::graph::modules::ModuleForwardResult>,
    /// Any hashes of params generated during the forward pass
    pub processed_params: Option<crate::graph::modules::ModuleForwardResult>,
    /// Any hashes of outputs generated during the forward pass
    pub processed_outputs: Option<crate::graph::modules::ModuleForwardResult>,
    /// max lookup input
    pub max_lookup_inputs: IntegerRep,
    /// max lookup input
    pub min_lookup_inputs: IntegerRep,
    /// max range check size
    pub max_range_size: IntegerRep,
    /// (optional) version of ezkl used
    pub version: Option<String>,
}

impl GraphWitness {
    ///
    pub fn get_float_outputs(&self, scales: &[crate::Scale]) -> Vec<Tensor<f32>> {
        self.outputs
            .iter()
            .enumerate()
            .map(|(i, x)| {
                x.iter()
                    .map(|y| (felt_to_f64(*y) / scale_to_multiplier(scales[i])) as f32)
                    .collect::<Tensor<f32>>()
            })
            .collect()
    }

    ///
    pub fn new(inputs: Vec<Vec<Fp>>, outputs: Vec<Vec<Fp>>) -> Self {
        GraphWitness {
            inputs,
            outputs,
            pretty_elements: None,
            processed_inputs: None,
            processed_params: None,
            processed_outputs: None,
            max_lookup_inputs: 0,
            min_lookup_inputs: 0,
            max_range_size: 0,
            version: None,
        }
    }

    /// Generate the rescaled elements for the witness
    pub fn generate_rescaled_elements(
        &mut self,
        input_scales: Vec<crate::Scale>,
        output_scales: Vec<crate::Scale>,
        visibility: VarVisibility,
    ) {
        let mut pretty_elements = PrettyElements {
            rescaled_inputs: self
                .inputs
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    let scale = input_scales[i];
                    t.iter()
                        .map(|x| dequantize(*x, scale, 0.).to_string())
                        .collect()
                })
                .collect(),
            inputs: self
                .inputs
                .iter()
                .map(|t| t.iter().map(|x| format!("{:?}", x)).collect())
                .collect(),
            rescaled_outputs: self
                .outputs
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    let scale = output_scales[i];
                    t.iter()
                        .map(|x| dequantize(*x, scale, 0.).to_string())
                        .collect()
                })
                .collect(),
            outputs: self
                .outputs
                .iter()
                .map(|t| t.iter().map(|x| format!("{:?}", x)).collect())
                .collect(),
            ..Default::default()
        };

        if let Some(processed_inputs) = self.processed_inputs.clone() {
            pretty_elements.processed_inputs = processed_inputs
                .get_result(visibility.input)
                .iter()
                // gets printed as hex string
                .map(|x| x.iter().map(|y| format!("{:?}", y)).collect())
                .collect();
        }

        if let Some(processed_params) = self.processed_params.clone() {
            pretty_elements.processed_params = processed_params
                .get_result(visibility.params)
                .iter()
                // gets printed as hex string
                .map(|x| x.iter().map(|y| format!("{:?}", y)).collect())
                .collect();
        }

        if let Some(processed_outputs) = self.processed_outputs.clone() {
            pretty_elements.processed_outputs = processed_outputs
                .get_result(visibility.output)
                .iter()
                // gets printed as hex string
                .map(|x| x.iter().map(|y| format!("{:?}", y)).collect())
                .collect();
        }

        self.pretty_elements = Some(pretty_elements);
    }

    ///
    pub fn get_polycommitments(&self) -> Vec<G1Affine> {
        let mut commitments = vec![];
        if let Some(processed_inputs) = &self.processed_inputs {
            if let Some(commits) = &processed_inputs.polycommit {
                commitments.extend(commits.iter().flatten());
            }
        }
        if let Some(processed_params) = &self.processed_params {
            if let Some(commits) = &processed_params.polycommit {
                commitments.extend(commits.iter().flatten());
            }
        }
        if let Some(processed_outputs) = &self.processed_outputs {
            if let Some(commits) = &processed_outputs.polycommit {
                commitments.extend(commits.iter().flatten());
            }
        }
        commitments
    }

    /// Export the ezkl witness as json
    pub fn as_json(&self) -> Result<String, GraphError> {
        let serialized = serde_json::to_string(&self).map_err(GraphError::from)?;
        Ok(serialized)
    }

    /// Load the model input from a file
    pub fn from_path(path: std::path::PathBuf) -> Result<Self, GraphError> {
        let file = std::fs::File::open(path.clone()).map_err(|e| {
            GraphError::ReadWriteFileError(path.display().to_string(), e.to_string())
        })?;

        let reader = std::io::BufReader::with_capacity(EZKL_BUF_CAPACITY, file);
        let witness: GraphWitness =
            serde_json::from_reader(reader).map_err(Into::<GraphError>::into)?;

        // check versions match
        crate::check_version_string_matches(witness.version.as_deref().unwrap_or(""));

        Ok(witness)
    }

    /// Save the model input to a file
    pub fn save(&self, path: std::path::PathBuf) -> Result<(), GraphError> {
        let file = std::fs::File::create(path.clone()).map_err(|e| {
            GraphError::ReadWriteFileError(path.display().to_string(), e.to_string())
        })?;
        // use buf writer
        let writer = std::io::BufWriter::with_capacity(EZKL_BUF_CAPACITY, file);

        serde_json::to_writer(writer, &self).map_err(|e| e.into())
    }

    ///
    pub fn get_input_tensor(&self) -> Vec<Tensor<Fp>> {
        self.inputs
            .clone()
            .into_iter()
            .map(|i| Tensor::from(i.into_iter()))
            .collect::<Vec<Tensor<Fp>>>()
    }

    ///
    pub fn get_output_tensor(&self) -> Vec<Tensor<Fp>> {
        self.outputs
            .clone()
            .into_iter()
            .map(|i| Tensor::from(i.into_iter()))
            .collect::<Vec<Tensor<Fp>>>()
    }
}

#[cfg(feature = "python-bindings")]
impl<'py> IntoPyObject<'py> for GraphWitness {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        // Create a Python dictionary
        let dict = PyDict::new(py);
        let dict_inputs = PyDict::new(py);
        let dict_params = PyDict::new(py);
        let dict_outputs = PyDict::new(py);

        let inputs: Vec<Vec<String>> = self
            .inputs
            .iter()
            .map(|x| x.iter().map(field_to_string).collect())
            .collect();

        let outputs: Vec<Vec<String>> = self
            .outputs
            .iter()
            .map(|x| x.iter().map(field_to_string).collect())
            .collect();

        dict.set_item("inputs", inputs).unwrap();
        dict.set_item("outputs", outputs).unwrap();
        dict.set_item("max_lookup_inputs", self.max_lookup_inputs)
            .unwrap();
        dict.set_item("min_lookup_inputs", self.min_lookup_inputs)
            .unwrap();
        dict.set_item("max_range_size", self.max_range_size)
            .unwrap();

        if let Some(processed_inputs) = &self.processed_inputs {
            //poseidon_hash
            if let Some(processed_inputs_poseidon_hash) = &processed_inputs.poseidon_hash {
                insert_poseidon_hash_pydict(&dict_inputs, processed_inputs_poseidon_hash).unwrap();
            }
            if let Some(processed_inputs_polycommit) = &processed_inputs.polycommit {
                insert_polycommit_pydict(&dict_inputs, processed_inputs_polycommit).unwrap();
            }

            dict.set_item("processed_inputs", dict_inputs).unwrap();
        }

        if let Some(processed_params) = &self.processed_params {
            if let Some(processed_params_poseidon_hash) = &processed_params.poseidon_hash {
                insert_poseidon_hash_pydict(&dict_params, processed_params_poseidon_hash).unwrap();
            }
            if let Some(processed_params_polycommit) = &processed_params.polycommit {
                insert_polycommit_pydict(&dict_params, processed_params_polycommit).unwrap();
            }

            dict.set_item("processed_params", dict_params).unwrap();
        }

        if let Some(processed_outputs) = &self.processed_outputs {
            if let Some(processed_outputs_poseidon_hash) = &processed_outputs.poseidon_hash {
                insert_poseidon_hash_pydict(&dict_outputs, processed_outputs_poseidon_hash)
                    .unwrap();
            }
            if let Some(processed_outputs_polycommit) = &processed_outputs.polycommit {
                insert_polycommit_pydict(&dict_outputs, processed_outputs_polycommit).unwrap();
            }

            dict.set_item("processed_outputs", dict_outputs).unwrap();
        }

        Ok(dict.into_any())
    }
}

#[cfg(feature = "python-bindings")]
fn insert_poseidon_hash_pydict(
    pydict: &Bound<'_, PyDict>,
    poseidon_hash: &Vec<Fp>,
) -> Result<(), PyErr> {
    let poseidon_hash: Vec<String> = poseidon_hash.iter().map(field_to_string).collect();
    pydict.set_item("poseidon_hash", poseidon_hash)?;

    Ok(())
}

#[cfg(feature = "python-bindings")]
fn insert_polycommit_pydict(
    pydict: &Bound<'_, PyDict>,
    commits: &Vec<Vec<G1Affine>>,
) -> Result<(), PyErr> {
    use crate::bindings::python::PyG1Affine;
    let poseidon_hash: Vec<Vec<PyG1Affine>> = commits
        .iter()
        .map(|c| c.iter().map(|x| PyG1Affine::from(*x)).collect())
        .collect();
    pydict.set_item("polycommit", poseidon_hash)?;

    Ok(())
}
