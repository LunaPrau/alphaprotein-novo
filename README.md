# AlphaProtein Novo

AlphaProtein Novo (AP Novo) is a generative diffusion pipeline for *de novo*
enzyme design via structural motif scaffolding.

This package includes a diffusion model which co-generates protein structures
and sequences conditioned on a catalytic motif and ligand context. This is run
using `run_generator.py`.

To form an end-to-end pipeline, the diffusion model is combined with optional
sequence redesign using LigandMPNN (`run_ligandmpnn.py`), structure prediction
using AlphaFold 3 (`run_alphafold.py`), and evaluation metrics
(`evaluate_design.py`). The `run_pipeline.py` script runs all stages in
sequence.

See the [Quickstart](#quickstart) section for a basic launch command and
[Example Design Campaigns](#example-design-campaigns) for more detailed
examples.

:ledger: **Note: Pretrained model weights are not part of this package and must
be downloaded from Google Cloud Storage (see
[Model Parameters](#model-parameters-weights) below).** Use is subject to these
[terms of use](WEIGHTS_TERMS_OF_USE.md). Scripts look for weights under
`./models/apnovo_generator` by default, or you can point to another directory
using `--apn_model_dir` (for `run_pipeline.py`), `--model_dir` (for
`run_generator.py`), or `settings.model_dir` in your design manifest.

## Installation & Environment Setup

Two Python environments are recommended:

-   **Primary Environment (`alphaprotein_novo`)**: Contains JAX. Runs diffusion
    generation, AlphaFold 3 folding, evaluation metrics, and the pipeline
    orchestrator.
-   **LigandMPNN Environment (`ligandmpnn`) [Optional]**: Contains PyTorch. Used
    to run LigandMPNN for sequence design.

### Primary Environment (`alphaprotein_novo`)

```bash
# Create and activate a Python 3.12 environment
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install JAX with CUDA 12 support (or CPU: uv pip install -U jax)
uv pip install -U "jax[cuda12]>=0.4.30"

# Install AlphaFold 3
uv pip install git+https://github.com/google-deepmind/alphafold3.git

# Install AP Novo in editable mode
cd /path/to/alphaprotein_novo
uv pip install -e .

# Compile AlphaFold 3 chemical component data (CCD pickle)
build_data
```

:ledger: **Tip: `build_data` compiles `ccd.pickle` and
`chemical_component_sets.pickle` required by AlphaFold 3 to parse ligands.** If
`components.cif` cannot be located, set the `LIBCIFPP_DATA_DIR` environment
variable to its directory before running `build_data`.

### LigandMPNN Environment (`ligandmpnn`) [Optional]

Only required if you plan to run optional sequence redesign with LigandMPNN:

```bash
# 1. Create and activate Python 3.11 environment
uv venv --python 3.11 .venv-ligandmpnn
source .venv-ligandmpnn/bin/activate

# 2. Clone LigandMPNN and download weights
git clone https://github.com/dauparas/LigandMPNN.git
cd LigandMPNN
bash get_model_params.sh ./model_params

# 3. Install dependencies
uv pip install -r requirements.txt
uv pip install "setuptools<82"  # ProDy requires pkg_resources removed in setuptools 82+

# 4. (Optional) Set environment variables. This allows you to avoid having to
# provide the --ligandmpnn_dir and --ligandmpnn_python flags to run_pipeline.py.
export LIGANDMPNN_DIR=/home/<your_username>/projects/LigandMPNN  # For example.
export LIGANDMPNN_PYTHON=/home/<your_username>/miniconda3/envs/ligandmpnn/bin/python  # For example.
```

### Model Parameters (Weights)

-   **AP Novo**: Download pretrained generator weights (`generator.bin.zst`)
    into `./models/apnovo_generator` (subject to the
    [AP Novo Parameters Terms of Use](WEIGHTS_TERMS_OF_USE.md); or configure a
    directory via `--apn_model_dir`, `--model_dir`, or `settings.model_dir` in
    your manifest):

```bash
mkdir -p models/apnovo_generator
wget -P models/apnovo_generator https://storage.googleapis.com/alphaprotein_novo/generator.bin.zst
```

-   **AlphaFold 3 Leaving Atom (AF3-LA)**: Download neural network weights
    (`af3_leaving_atom.bin.zst`) into `./models/af3_la` (subject to the
    [AlphaFold 3 Parameters Terms of Use](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md)).
    Structure prediction uses these by default, because AF3-LA is fine-tuned for
    leaving atom handling and therefore models the covalent intermediates common
    in AP Novo designs:

```bash
mkdir -p models/af3_la
wget -P models/af3_la https://storage.googleapis.com/alphafold3/af3_leaving_atom.bin.zst
```

Alternatively, stock **AlphaFold 3** weights (`af3.bin.zst`) can be used,
subject to the same terms of use. Pass `--model_dir=./models/af3` (or
`--af3_model_dir=./models/af3` to `run_pipeline.py`) to select them:

```bash
mkdir -p models/af3
wget -P models/af3 https://storage.googleapis.com/alphafold3/af3.bin.zst
```

## Quickstart

The standard way to run AP Novo is via `run_pipeline.py`. It executes design
generation, sequence design, folding, and metrics calculation end-to-end from a
single "manifest" file containing configuration settings:

```bash
# Ensure the primary environment is active
source .venv/bin/activate

python run_pipeline.py \
  --manifest=examples/kemp_eliminase/kemp_manifest.json \
  --output_dir=./kemp_campaign \
  --af3_model_dir=./models/af3_la \
  --ligandmpnn_dir=/path/to/LigandMPNN \
  --ligandmpnn_python="/path/to/ligandmpnn/.venv-ligandmpnn/bin/python"
```

:ledger: **Tip: To quickly verify your installation without waiting for a
full-length diffusion trajectory, swap in
`examples/kemp_eliminase/kemp_test_manifest.json`**, which runs the same Kemp
eliminase benchmark with reduced sampling steps.

<!-- disableFinding(LINK_ID) -->

Outputs are organized into stage-specific subdirectories under `--output_dir`
(see [Generated Artifacts & Reports](#generated-artifacts--reports) below). By
default, re-running the pipeline resumes interrupted campaigns by skipping
completed designs; you can also force a full rerun or execute individual stages
(see [Modular Stage Execution](#modular-stage-execution)).

<!-- enableFinding(LINK_ID) -->

## Design Manifests

Campaigns are configured using a JSON manifest describing the design problem,
motif conditioning, and downstream processing.

Here is an example based on `examples/kemp_eliminase/kemp_manifest.json`:

```json
{
  "settings": {
    "model_dir": "./models/apnovo_generator"
  },
  "defaults": {
    "input_file": "kemp_eliminase_motif.cif",
    "is_author_naming": true,
    "num_sampling_steps": 1000
  },
  "designs": [
    {
      "name": "kemp_tight",
      "motif_str": "A1,A2,A3|10-40,{},2-30,{},2-30,{},10-40/B1",
      "motif_atoms": "A1:NE2,ND1 A2:OD1 A3:ND2",
      "num_designs": 200
    }
  ],
  "resequence": {
    "enabled": true,
    "temperature": 0.1
  },
  "folding": {
    "inputs": ["resequenced"],
    "seeds": [230],
    "states": [
      {"name": "monomer", "ligands": []},
      {"name": "complex", "ligands": [{"id": "B", "ccd_code": "6NT"}]}
    ]
  },
  "evaluation": {
    "suite": "kemp_eliminase"
  }
}
```

See [below](#example-design-campaigns) for further, full-fledged examples.

### Manifest Structure

-   **`settings`**: Run-level configuration (e.g. `model_dir`, `output_dir`).
    Overridden by corresponding CLI flags.
-   **`defaults`**: Default fields applied to any design job that does not
    explicitly set them (e.g. `num_sampling_steps`, `seed_start`).
-   **`designs`**: List of design tasks. Each specifies the number of designs
    and the input motif (see [Design Inputs](#design-inputs) below).
-   **`resequence`**: Configures LigandMPNN sequence redesign (`enabled`,
    `temperature`). Set `"enabled": false` to fold the generated sequence
    directly without LigandMPNN.
-   **`folding`**: Specifies which structures to fold (`inputs: ["generated"]`
    or `["resequenced"]`), RNG `seeds`, and target `states` (e.g. apo monomer,
    ligand complex).
-   **`evaluation`**: Sets evaluation parameters, including the enzyme
    evaluation `suite` (`"kemp_eliminase"`, `"serine_esterase"`,
    `"dehp_esterase"`, `"carbene_transfer"`, or `"nitrene_transfer"`) and an
    optional `reference_cif` override (by default, each design is scored against
    its own per-sample ground-truth motif structure).

:ledger: **Note: Relative file paths in the manifest resolve relative to the
directory containing the manifest file.**

### Design Inputs

De novo enzyme design starts with a 3D arrangement of sidechain and ligand atoms
that represents a transition-state or intermediate of the target reaction. The
goal of design is to generate a protein that holds this motif in the intended
conformation.

At a high level, you will need to decide the following:

-   Input motif: what are the catalytic residues and small molecule ligands that
    represent your reaction? What conformation should the enzyme bind them in?
-   Full or partial residue motif: Do you want to constrain the backbone
    positions of all residues in the motif, or just functional groups on the
    sidechains?
-   Indexed or unindexed motif residues: Do you want to specify the residue
    indexes (primary sequence positions) of motif residues on the design ahead
    of generation (indexed), or have the model decide where they go (unindexed)?
    Unindexed conditioning can in theory find more optimal motif placements, but
    we find that indexed conditioning gives slightly better results in practice.
-   Novel scaffold or partial diffusion: Do you want to generate a protein from
    scratch, or re-use an existing protein structure as a starting point
    (partial diffusion)?

The design problem is specified in full using the following fields in the
`designs` section of the manifest:

-   `input_file` is a path to a CIF file containing the input motif.
-   `motif_str` is a string describing which parts of the input structure make
    up the motif and how they are arranged in the output design.
    -   The string specifies a series of design chains delimited by `/`.
    -   The first design chain contains a series of comma-separated segments
        representing scaffolded motif or designable regions. Subsequent chains
        should consist of a single residue, representing a (fixed) ligand.
    -   For example, `A1,5-10,A5-6,3/B1` means "generate a design that scaffolds
        residue 1 of chain A of the input structure, followed by 5 to 10
        residues of generated protein, followed by scaffolding residues 5-6 of
        input chain A, followed by 3 residues of generated protein, with a 2nd
        chain consisting exactly of the atoms in input chain B (a ligand)".
    -   In this example, residues 1, 5, and 6 of input chain A and 1 of chain B
        are "fixed" to their input coordinates, and the model will try to
        generate coordinates for the remaining residues to best preserve the
        position of the fixed residues or ligands.
    -   Designable length ranges are sampled before generation, e.g. for `5-10`,
        a number between 5 and 10 (inclusive) is chosen uniformly at random and
        fixed for the duration of the diffusion process.
    -   The ordering of the motif segments in `motif_str` can be sampled by
        writing `"A1,A5-6|{},5-10,{},3/B1"`. This will resolve to either
        `A1,5-10,A5-6,3/B1` or `A5-6,5-10,A1,3/B1` with equal probability.
-   `seq_length` is an optional string constraining the total residue length of
    the designed protein chain (either an exact integer like `"150"` or an
    inclusive range like `"120-160"`). When specified, designable segment
    lengths in `motif_str` are sampled conditioned on the total chain length
    falling within this range.
-   `unindexed_motif_residues` is an optional string containing motif residues
    for "unindexed" conditioning. By default, the motif residues are at fixed,
    pre-defined positions along the primary sequence ("indexed" conditioning).
    Under unindexed conditioning, the model decides during generation where the
    motif should go.
    -   This is a comma-separated list of residues or residue ranges, e.g.
        `"A1,A2,A3"` or `"A120-121,A188"`.
    -   If specified, `motif_str` must contain a single designable element (e.g.
        `10-250/B1`), with no fixed residues specified other than ligand chains
        (which are always fixed).
-   `motif_atoms` is an optional string describing which atoms of motif residues
    are considered fixed, allowing for side chains to be generated by the model
    (and thereby be "flexible"). This is sometimes called "tip atom" or "atomic
    motif" conditioning.
    -   This is a space-separated list of groups, where each group consists of a
        residue name and a comma-separated list of atom names.
    -   If not specified, all atoms of the residues in `motif_str` are fixed.
    -   If specified, the indicated atoms are fixed and the rest of the residue
        is generated by the model.
    -   For example, `"A1:NE2,ND1 A2:OD1"` means "fix NE2 and ND1 of residue A1
        and OD1 of residue A2".
-   `reseq_residues` is an optional string specifying residues on the motif to
    "resequence", or allow to change in amino-acid identity during the
    generation and resequence stages. This can be used to constrain the backbone
    position of a residue while allowing its sidechain to vary in identity.
-   `is_author_naming` is a boolean indicating whether the motif string uses
    author-assigned residue numbers or PDB "internal" numbering. PyMOL and the
    literature almost always use author naming. The PDB structure viewer uses
    internal numbering.
-   `partial_diffusion_input_file` is an optional path to a CIF file containing
    a starting protein structure (such as a parent design) to diversify via
    partial diffusion. Must be specified together with
    `partial_diffusion_num_steps`.
-   `partial_diffusion_num_steps` is an optional integer specifying how many
    reverse diffusion steps (out of `num_sampling_steps`) to unroll when running
    partial diffusion from `partial_diffusion_input_file`. Setting this to 600
    will lead to outputs with roughly TM-score 95 to the parent design; 900
    steps will lead to TM-score ~80. (Total number of denoising steps is 1000.)

### Example Design Campaigns

The `examples/` directory contains manifest files and input motif structures
(`*.cif`) that partially reproduce the settings used for the best designs in the
AP Novo paper. Problem-specific evaluation metrics are also included in the code
for each of these examples, and are invoked by the manifests using the
`evaluation.suite` field. Note that this repo is a port of the original
(Google-internal) pipeline used to generate the designs in the paper. Some
settings (e.g. numbers of resequences and folding seeds) have been reduced
relative to the paper so that example runs can complete in a reasonable time.

Run any of the example campaigns end-to-end using, for example:

```bash
python run_pipeline.py \
  --manifest=examples/kemp_eliminase/kemp_manifest.json \
  --output_dir=/tmp/apn_example_kemp
```

This assumes that AP Novo and AF3 weights are in the default locations and
LigandMPNN environment variables are set.

Example manifests:

1.  **Kemp eliminase (`examples/kemp_eliminase/`)**

    -   Motif contains glutamate catalytic base and serine oxyanion hole H-bond
        donor with 6-nitrobenzotriazole transition-state analog. Used for the
        design of `GDM_KE_1483`.
    -   Main pipeline: `kemp_manifest.json`. Unindexed conditioning variant:
        `kemp_unindexed_manifest.json`. Fast testing variant with 50 denoising
        steps (will not produce good designs): `kemp_test_manifest.json`.
    -   Folds monomer (apo) and 6NT-bound states with 5 AF3 seeds each.
    -   Computes self-consistency, pocket RMSD, and Kemp eliminase catalytic
        geometry metrics.

2.  **4MU-Ac serine esterase (`examples/serine_esterase/`)**

    -   Motif is derived from cutinase 1xzm, with a Ser-His-Asp catalytic triad,
        three oxyanion hole H-bond donors, and 4-methylumbelliferyl acetate
        tetrahedral intermediate. Used for the design of `GDM_SE_2937`.
    -   AF3 folding across all five reaction states (`monomer`, `es`, `complex`
        / TI1, `aei`, and `ti2`), with evaluation of catalytic triad/oxyanion
        hole H-bonds and cross-state (ES/TI1) ligand RMSD and coordinate
        standard deviation.

3.  **DEHPase — novel scaffold (`examples/dehp_esterase_denovo/`)**

    -   Motif is derived from kexin 1r64, with a Ser-His-Asp catalytic triad, 2
        oxyanion hole H-bond donors, and bis(2-ethylhexyl) phthalate tetrahedral
        intermediate, used to design `GDM_DEHP_0176`.
    -   AF3 folding in 5 reaction states as in 4MU-Ac esterase example.

4.  **DEHPase — partial diffusion
    (`examples/dehp_esterase_partial_diffusion/`)**

    -   Partial-diffusion (backtracking 575 steps, out of 1000) from parent
        design `GDM_DEHP_0176.cif`, used to design `GDM_DEHP_0376`.
    -   Same folding and evaluation setup as in 4MU-Ac and novel-scaffold
        DEHPase examples above.

5.  **Carbene transferase (`examples/carbene_transferase/`)**

    -   Motif with axial histidine, heme cofactor (`HEM`), and
        (1S,2S)-cyclopropanation transition-state/product conformer, used to
        create `GDM_CT_0103`.
    -   Folding in 2 states: `monomer` (apo) and `complex` (`HEM` + product).
        Evaluation computes self-consistency, pocket-aligned ligand RMSD, and
        carbene transferase heme-pocket geometry metrics.

6.  **Piperidine synthase / nitrene transferase
    (`examples/nitrene_transferase/`)**

    -   Motif with axial histidine and heme-substrate cofactor complex, used to
        create `GDM_NT_0151`.
    -   Folding in 7 states: `monomer` (apo), `complex`, `heme_substrate`,
        `heme_piperidine_R`, `heme_pyrrolidine_S`,
        `fiveazidopentylbenzene_heme_1`, `fiveazidopentylbenzene_heme_2`.
        Evaluation computes cross-state Fe/N self-RMSD, Fe–N distances, and
        regioselectivity based on near-attack atom distances
        (`piperidine_propensity`).

:ledger: **Note: `use_side_chain_context`**: By default, `run_ligandmpnn.py` and
`resequence.use_side_chain_context` enable fixed-residue side-chain atom context
(`--ligand_mpnn_use_side_chain_context=1`), which improves design outcomes. In
these example manifests, `"use_side_chain_context": false` is set to reflect the
exact settings used in the paper, but for your own designs you should generally
enable side-chain context.

## Generated Artifacts & Reports

`run_pipeline.py` organizes outputs by pipeline stage:

```text
kemp_campaign/
├── pipeline_index.json         # Campaign manifest snapshot, hash, and run metadata
├── logs/                       # Per-stage stdout and stderr logs
│   ├── generation.log
│   ├── resequence.log
│   ├── folding.log
│   └── evaluation.log
├── metadata/                   # Run-level per-design metadata (<prefix>_metadata.json) and designs.json
├── 01_generation/              # Stage 1: Generated structures (.cif/.pdb), motifs (_motif.cif), and sequences (.fa)
├── 02_resequence/              # Stage 2: Resequenced structures (_reseq.cif) and sequences (.fa)
├── 03_folded/                  # Stage 3: AlphaFold 3 predicted models, inputs, and confidence JSONs
└── 04_eval/                    # Stage 4: Per-design evaluation JSONs and evaluation_summary.csv
```

All files generated in Stage 1 share the prefix `<prefix>`
(`<job>_<design_num>`, e.g. `kemp_0000`). When sequence redesign is enabled,
Stage 2 appends `_seq<NN>` (e.g. `kemp_0000_seq00`), which becomes the design
prefix used in Stages 3 and 4.

### Per-Design Artifacts

<!-- mdformat off(prevent table wrapping) -->

| Stage | Artifact | Description |
| :--- | :--- | :--- |
| **Metadata** | `metadata/<prefix>_metadata.json` | Run-level design metadata (fixed residues, seeds, motif paths, and folding states). |
| **Generation** | `<prefix>.cif`, `<prefix>.pdb` | Generated backbone structure with ligand coordinates. |
| | `<prefix>_motif.cif` | Ground-truth motif structure re-indexed to match the generated design. |
| | `<prefix>.fa` | Single-letter *de novo* amino acid sequence. |
| **Resequence** | `<prefix>_seq<NN>_reseq.cif` | Scaffold structure with LigandMPNN-redesigned sequence. |
| | `<prefix>_seq<NN>.fa`, `seqs/<prefix>.fa` | Individual and combined LigandMPNN-redesigned sequences. |
| **Folding** | `<prefix>_<state>_folded_seed<N>.cif` | AlphaFold 3 predicted structure for the specified state and seed index `<N>`. |
| | `<prefix>_<state>_confidences_seed<N>.json` | AlphaFold 3 confidence scores (`plddt`, `ptm`, `iptm`, `ranking_confidence`, `per_atom_plddt`, `chain_pair_pde_mean`) and the `seed` used for folding. |
| | `<prefix>_<state>_af3_input.json` | AlphaFold 3 folding input JSON for the specified state across all configured folding seeds. |
| **Evaluation** | `<prefix>_evaluation.json` | Flat dictionary of metrics (`metrics`) for the individual design across folded states and seeds. |

<!-- mdformat on -->

### Campaign Summary Reports

-   `04_eval/evaluation_summary.csv`: Tabular spreadsheet written by Stage 4
    with one row per design containing all scalar aggregated and per-seed
    metrics across states.
-   `metadata/designs.json`: Index written by Stage 1 mapping generated designs
    to seeds, prefixes, and completion status.
-   `pipeline_index.json`: Campaign-level record written by `run_pipeline.py`
    with the manifest path, SHA-256 hash, completed stages, and design prefixes.

## Modular Stage Execution

`run_pipeline.py` tracks completed work and supports partial or stage-by-stage
execution, and each stage script can also be invoked directly.

### Resume & Stage Controls (`run_pipeline.py`)

-   **Resuming interrupted runs**: By default (`--resume=true`), re-running the
    pipeline command skips designs that have already finished in `--output_dir`.
-   **Forced rerun**: Pass `--noresume` to rerun all stages from scratch.
-   **Start from specific stage**: Pass `--from_stage` to resume execution
    starting from a specific stage (`generation`, `resequence`, `folding`,
    `evaluation`).
-   **Run single stage**: Pass `--only_stage=folding` to run only that stage
    against existing outputs on disk.

### Multi-GPU & Multi-CPU Parallelization

The pipeline automatically scales across available hardware on a single node or
across manually sharded workers:

-   **Multi-GPU Sharding (Stages 1 & 3)**: When multiple GPUs are visible via
    `CUDA_VISIBLE_DEVICES` (or detected via `nvidia-smi`), `run_pipeline.py`—as
    well as standalone invocations of `run_generator.py` and
    `run_alphafold.py`—automatically spawns one worker process per GPU
    (`--worker_id=w --num_workers=N` with `CUDA_VISIBLE_DEVICES` pinned to each
    device) and distributes pending designs across workers. To restrict which
    GPUs are used, set `CUDA_VISIBLE_DEVICES` (e.g. `CUDA_VISIBLE_DEVICES=0,1`).
-   **Multi-CPU Parallelism (Stages 2 & 4)**:
    -   `run_ligandmpnn.py` runs LigandMPNN on CPU (`CUDA_VISIBLE_DEVICES=""`)
        and shards pending designs across `--num_workers` parallel CPU processes
        (default: `min(16, os.cpu_count())`), using the same worker pool to
        write resequenced mmCIF structures in parallel.
    -   `evaluate_design.py` evaluates folded designs in parallel across up to
        `min(24, os.cpu_count())` CPU worker processes when `--num_workers=1`.
-   **Manual / Multi-Node Sharding**: You can also shard Stages 1, 3, or 4
    explicitly across separate jobs or nodes by passing `--worker_id=<0..N-1>`
    and `--num_workers=<N>` directly to `run_generator.py`, `run_alphafold.py`,
    or `evaluate_design.py`.

### Stage 1: De Novo Generation (`run_generator.py`)

Generates protein backbones and initial sequences from a manifest:

```bash
python run_generator.py \
  --manifest=examples/kemp_eliminase/kemp_manifest.json \
  --output_dir=./samples
```

<!-- mdformat off(prevent table wrapping) -->

| Key Flag | Description |
| :--- | :--- |
| `--manifest` | (Required) Path to design manifest JSON. |
| `--output_dir` | Directory where generated structures, sequences, and metadata are written. |
| `--model_dir` | Model weights directory (defaults to `./models/apnovo_generator`). |
| `--resume` | Skip already completed designs in `--output_dir` (default: `true`). |
| `--return_all_data` | Include unrolled diffusion trajectory diagnostics (default: `false`). |
| `--worker_id` | 0-indexed worker ID for multi-GPU sharding (default: `0`). |
| `--num_workers` | Total number of parallel workers for sharding (default: `1`; auto-spawned across visible GPUs when `1`). |

<!-- mdformat on -->

*Run `python run_generator.py --help` for all options.*

### Stage 2: Sequence Redesign (`run_ligandmpnn.py`) [Optional]

Resequences generated backbones with LigandMPNN from within the primary
environment:

```bash
python run_ligandmpnn.py \
  --input_dir=./samples \
  --output_dir=./samples \
  --ligandmpnn_dir=/path/to/LigandMPNN \
  --python_executable="/path/to/ligandmpnn/.venv-ligandmpnn/bin/python"
```

<!-- mdformat off(prevent table wrapping) -->

| Key Flag | Description |
| :--- | :--- |
| `--input_dir` | Directory holding designs to resequence (located via design metadata). |
| `--output_dir` | Destination directory for resequenced structures and FASTA files. |
| `--ligandmpnn_dir` | Path to cloned LigandMPNN repository. |
| `--python_executable` | Python interpreter from the `ligandmpnn` environment. |
| `--temperature` | Sampling temperature for sequence generation (default: `0.1`). |
| `--checkpoint` | Model checkpoint (default: `ligandmpnn_v_32_030_25.pt`). |
| `--ligand_mpnn_use_side_chain_context` | Enables side-chain context for fixed residues (`resequence.use_side_chain_context`: `true` by default). |
| `--num_workers` | Number of parallel CPU workers for LigandMPNN inference and structure post-processing (default: `min(16, os.cpu_count())`). |

<!-- mdformat on -->

*Run `python run_ligandmpnn.py --help` for all options.*

### Stage 3: Structure Prediction (`run_alphafold.py`)

Predicts structures using AlphaFold 3. Folding runs against the AF3-LA weights
with `--fix_standalone_glycans=true`, so that leaving atoms on covalent
intermediates and unbonded glycan ligands are modelled rather than stripped:

```bash
python run_alphafold.py \
  --input_dir=./samples \
  --output_dir=./folded_outputs \
  --model_dir=./models/af3_la
```

<!-- mdformat off(prevent table wrapping) -->

| Key Flag | Description |
| :--- | :--- |
| `--input_dir` | Directory containing designs to fold. |
| `--output_dir` | Destination directory for predicted structures and confidences. |
| `--model_dir` | Directory containing AlphaFold 3 weights (defaults to `models/af3_la`, holding `af3_leaving_atom.bin.zst`). |
| `--fix_standalone_glycans` | Preserve leaving atoms on unbonded ("standalone") glycan ligands (default: `true`). |
| `--input_structure` | Structure to fold: `generated` (default) or `resequenced`. |
| `--seed` | Random seed for AlphaFold 3 inference (default: `230`). |
| `--resume` | Skip designs already folded into `--output_dir` (default: `true`). |
| `--worker_id` | 0-indexed worker ID for multi-GPU sharding (default: `0`). |
| `--num_workers` | Total number of parallel workers for sharding (default: `1`; auto-spawned across visible GPUs when `1`). |

<!-- mdformat on -->

*Run `python run_alphafold.py --help` for all options.*

### Stage 4: Design Evaluation (`evaluate_design.py`)

Computes various structural, geometric, and self-consistency metrics:

```bash
python evaluate_design.py \
  --input_dir=./folded_outputs \
  --output_dir=./eval_outputs
```

<!-- mdformat off(prevent table wrapping) -->

| Key Flag | Description |
| :--- | :--- |
| `--input_dir` | Directory containing folded structures and metadata. Repeatable. |
| `--output_dir` | Directory where `<prefix>_evaluation.json` and `evaluation_summary.csv` will be written. |
| `--eval_reference_cif` | Optional path to ground-truth motif CIF (overrides metadata). |
| `--resume` | Reuse evaluation for designs already in `--output_dir` (default: `true`). |
| `--worker_id` | 0-indexed worker ID when sharding across multiple evaluation jobs (default: `0`). |
| `--num_workers` | Total number of shard workers when sharding across multiple evaluation jobs (default: `1`; single-worker runs auto-parallelize across up to `min(24, os.cpu_count())` CPU processes). |

<!-- mdformat on -->

*Run `python evaluate_design.py --help` for all options.*

## Evaluation Metrics Reference

`evaluate_design.py` computes key metrics across each folded state (see
`metrics/` for complete implementations):

-   **Self-Consistency**:
    -   `rmsd`: $C_\alpha$ RMSD between the predicted model and designed
        scaffold (Å).
    -   `tm_score`: Template Modeling score between predicted and designed
        structures (0 to 1).
    -   `lddt` & `gdt_ha`: Local Distance Difference Test and High-Accuracy
        Global Distance Test scores.
-   **Catalytic Motif Preservation**:
    -   `motif_allatom_rmsd`: All-atom RMSD of catalytic motif residues with
        side-chain permutation symmetry.
    -   `motif_bb_aligned_allatom_rmsd`: All-atom motif RMSD after aligning
        active-site backbones.
-   **Ligand Pocket Geometry**:
    -   `mean_pocket_bb_aligned_ligand_rmsd`: Ligand RMSD after aligning pocket
        backbone atoms. Reference-dependent ligand consistency metrics
        (`mean_pocket_bb_aligned_ligand_rmsd`,
        `pocket_bb_aligned_ligand_rmsd/<ligand>`, `mean_pocket_bb_rmsd`,
        `motif_with_ligand_allatom_rmsd`, and `interface_lddt`) are only
        computed for folding states whose ligand heavy-atom set (`(chain_id,
        res_id, res_name, atom_name)`) and intra-residue covalent bond graph
        match the ligand in the generated structure; they are omitted for states
        folded with different ligands or atom-numbering topologies (e.g. cleaved
        intermediates `aei`/`ti2` or `4MU-Ac` `es`). When a non-covalent
        substrate state and covalent transition state share the same heavy-atom
        names and bond connectivity (such as DEHP esterase `es` vs. `ti1`),
        these ligand consistency metrics are computed for both.
    -   `percent_ligand_bb_clashes_1_5`: Percentage of ligand atoms clashing
        with protein backbone (< 1.5 Å). Computed for all ligand-containing
        folding states.
-   **AlphaFold 3 Confidence**:
    -   `plddt`: Mean per-atom predicted lDDT (0 to 100).
    -   `ptm` & `iptm`: Predicted TM-score and Interface predicted TM-score.

## Licensing & Disclaimer

Copyright 2026 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you
may not use this file except in compliance with the Apache 2.0 license. You may
obtain a copy of the Apache 2.0 license at:
[https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0)

The AlphaProtein Novo Generator model parameters are made available under the
[AlphaProtein Novo Generator Model Parameters Terms of Use](https://github.com/google-deepmind/alphaprotein-novo/blob/main/WEIGHTS_TERMS_OF_USE.md)
(the "APN Terms"); you may not use these except in compliance with the Terms.
You may obtain a copy of the Terms at
[https://github.com/google-deepmind/alphaprotein-novo/blob/main/WEIGHTS_TERMS_OF_USE.md](https://github.com/google-deepmind/alphaprotein-novo/blob/main/WEIGHTS_TERMS_OF_USE.md).

Any pre-computed outputs of AlphaProtein Novo Generator in this repository will
be subject to the
[AlphaProtein Novo Generator Outputs Terms of Use](https://github.com/google-deepmind/alphaprotein-novo/blob/main/OUTPUT_TERMS_OF_USE.md)
(the “Output Terms”), you may not use any output except in compliance with the
Output Terms. You may obtain a copy of the Output Terms at
[https://github.com/google-deepmind/alphaprotein-novo/blob/main/OUTPUT_TERMS_OF_USE.md](https://github.com/google-deepmind/alphaprotein-novo/blob/main/OUTPUT_TERMS_OF_USE.md).

The AlphaFold 3 Leaving Atom model parameters are made available under the
[AlphaFold 3 Model Parameters Terms of Use](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md)
(the "Terms"); you may not use these except in compliance with the Terms. You
may obtain a copy of the Terms at
[https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md).

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
[https://creativecommons.org/licenses/by/4.0/legalcode](https://creativecommons.org/licenses/by/4.0/legalcode)

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0, the APN Terms, the Output
Terms, the Terms or CC-BY licenses are distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
licenses for the specific language governing permissions and limitations under
those licenses.

You are solely responsible for determining the appropriateness of using the
software, model parameters, materials or using or distributing outputs, and
assume any and all risks associated with such use or distribution and your
exercise of rights and obligations under these Terms. You and anyone you share
output with are solely responsible for these and their subsequent uses.

Output are predictions with varying levels of confidence and should be
interpreted carefully. Use discretion before relying on, publishing, downloading
or otherwise using AlphaProtein Novo Generator or AlphaFold 3 Leaving Atom.

All software, model parameters, materials and any outputs you create are for
theoretical modeling only. They are not intended, validated, or approved for
clinical use. You should not use the software, model parameters, materials or
outputs for clinical purposes or rely on them for medical or other professional
advice. Any content regarding those topics is provided for informational
purposes only and is not a substitute for advice from a qualified professional.

This is not an official Google product.
