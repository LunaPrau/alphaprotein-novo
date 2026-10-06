# Copyright 2026 DeepMind Technologies Limited
#
# AlphaProtein Novo source code is licensed under the Apache License,
# Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""LigandMPNN sequence redesign runner for AlphaProtein Novo.

Executes optional sequence redesign on generated protein-ligand backbones using
LigandMPNN. AlphaProtein Novo generates complete sequences directly; this step
is optional if additional sequence optimization is desired.

Typical workflow:
  1. Generate samples with AlphaProtein Novo:
     python run_generator.py --output_dir=/tmp/samples

  2. Run sequence redesign using the LigandMPNN Python interpreter (optional):
     python run_ligandmpnn.py \
       --input_dir=/tmp/samples \
       --ligandmpnn_dir=/path/to/LigandMPNN \
       --python_executable=/path/to/conda/envs/ligandmpnn/bin/python \
       --output_dir=/tmp/samples

  3. In your AlphaFold 3 environment, predict structures:
     python run_alphafold.py \
       --input_dir=/tmp/samples \
       --output_dir=/tmp/folded_out \
       --model_dir=/path/to/alphafold3_weights
"""

from collections.abc import Sequence
import concurrent.futures
import dataclasses
import json
import os
import subprocess
import sys
from typing import Any

from absl import app
from absl import flags
from absl import logging
from alphafold3 import structure
from alphaprotein_novo.data import design_manifest
from alphaprotein_novo.data import structure_utils
from etils import epath

# Input / Output flags.
_INPUT_DIR = epath.DEFINE_path(
    'input_dir',
    None,
    'Output folder of run_generator.py, containing generated structures.',
)
_OUTPUT_DIR = epath.DEFINE_path(
    'output_dir',
    epath.Path('./ligandmpnn_outputs'),
    'Output directory where LigandMPNN writes redesigned sequences and scores.',
)

# LigandMPNN installation and model checkpoint flags.
_LIGANDMPNN_DIR = epath.DEFINE_path(
    'ligandmpnn_dir',
    epath.Path(os.environ.get('LIGANDMPNN_DIR', './LigandMPNN')),
    'Path to the cloned dauparas/LigandMPNN repository directory.',
)
_CHECKPOINT = flags.DEFINE_string(
    'checkpoint',
    'ligandmpnn_v_32_030_25.pt',
    'Model checkpoint filename or full path. If a relative filename is given,'
    ' resolved against <ligandmpnn_dir>/model_params/.',
)
_MODEL_TYPE = flags.DEFINE_string(
    'model_type',
    'ligand_mpnn',
    'LigandMPNN model type (e.g. ligand_mpnn, protein_mpnn).',
)

# Sampling & redesign hyperparameters.
_TEMPERATURE = flags.DEFINE_float(
    'temperature',
    0.1,
    'Sampling temperature for sequence generation.',
)
_FIXED_RESIDUES = flags.DEFINE_string(
    'fixed_residues',
    None,
    'Space-separated list of fixed residue tokens (e.g. "A12 A45 A88").'
    ' If not provided, automatically loaded from metadata.json.',
)
_USE_ATOM_CONTEXT = flags.DEFINE_integer(
    'ligand_mpnn_use_atom_context',
    1,
    'Whether to use small-molecule ligand atom coordinates (1=enabled,'
    ' 0=disabled).',
)
_CUTOFF_FOR_SCORE = flags.DEFINE_float(
    'ligand_mpnn_cutoff_for_score',
    8.0,
    'Cutoff distance in Angstroms for score residue selection.',
)
_USE_SIDE_CHAIN_CONTEXT = flags.DEFINE_integer(
    'ligand_mpnn_use_side_chain_context',
    1,
    'Whether to use side-chain atom context for fixed residues (1=enabled,'
    ' 0=disabled).',
)
_OMIT_AA = flags.DEFINE_string(
    'omit_AA',
    'X',
    'Amino acid types to omit from design (e.g. "X" or "C").',
)
_NUM_SEQUENCES = flags.DEFINE_integer(
    'num_sequences',
    1,
    'Number of redesigned sequences per design.',
)
_PYTHON_EXECUTABLE = flags.DEFINE_string(
    'python_executable',
    os.environ.get('LIGANDMPNN_PYTHON') or sys.executable,
    'Python interpreter executable to invoke LigandMPNN with.',
)

_DEFAULT_CPU_WORKERS = min(16, os.cpu_count() or 1)

# Script behavior flags. Do not affect the designs themselves.
_RESUME = flags.DEFINE_bool(
    'resume',
    True,
    'Whether to skip designs whose resequenced structure already exists in'
    ' output_dir, allowing an interrupted batch to be continued.',
)
_NUM_WORKERS = flags.DEFINE_integer(
    'num_workers',
    _DEFAULT_CPU_WORKERS,
    'Number of parallel CPU workers for running LigandMPNN and post-processing'
    ' redesigned structures.',
)


def load_metadata(metadata_path: epath.PathLike) -> dict[str, Any]:
  """Loads sample metadata JSON dictionary."""
  path = epath.Path(metadata_path)
  if not path.exists():
    raise FileNotFoundError(f'Metadata file not found at: {path}')
  return json.loads(path.read_text())


@dataclasses.dataclass(frozen=True, kw_only=True)
class DesignInputs:
  """Resolved structures, metadata, and fixed residues for one design."""

  pdb_path: epath.Path
  metadata_path: epath.Path
  cif_path: epath.Path
  fixed_residues: str | None


def _resolve_designs(
    *,
    input_dir: epath.PathLike | None = None,
    fixed_residues_override: str | None = None,
) -> list[DesignInputs]:
  """Returns the resolved inputs of every design in `input_dir`."""
  if input_dir is None:
    raise ValueError('Input directory must be specified via --input_dir.')
  in_dir = epath.Path(input_dir)
  meta_dir = design_manifest.metadata_dir(in_dir)
  metadata_paths = design_manifest.find_all_design_metadata(meta_dir)
  in_dir_names = design_manifest.list_dir_names(in_dir)

  if fixed_residues_override is not None:
    loaded_metadata = [{} for _ in metadata_paths]
  elif len(metadata_paths) > 4:
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(16, len(metadata_paths))
    ) as pool:
      loaded_metadata = list(pool.map(load_metadata, metadata_paths))
  else:
    loaded_metadata = [load_metadata(mp) for mp in metadata_paths]

  designs: list[DesignInputs] = []
  for metadata_path, metadata in zip(metadata_paths, loaded_metadata):
    prefix = design_manifest.design_prefix(metadata_path)
    pdb_name = f'{prefix}{design_manifest.PDB_SUFFIX}'
    pdb_path = in_dir / pdb_name
    if pdb_name not in in_dir_names:
      raise FileNotFoundError(f'Input PDB structure not found at {pdb_path}.')
    cif_name = f'{prefix}{design_manifest.CIF_SUFFIX}'
    cif_path = in_dir / cif_name
    if cif_name not in in_dir_names:
      raise FileNotFoundError(f'Base mmCIF structure not found at {cif_path}.')
    fixed_residues = (
        fixed_residues_override
        if fixed_residues_override is not None
        else metadata.get('fixed_residues_flag')
    )
    designs.append(
        DesignInputs(
            pdb_path=pdb_path,
            metadata_path=metadata_path,
            cif_path=cif_path,
            fixed_residues=fixed_residues,
        )
    )
  return designs


def resolve_checkpoint(
    ligandmpnn_dir: epath.PathLike,
    checkpoint: str,
) -> epath.Path:
  """Resolves the model checkpoint path."""
  chk_path = epath.Path(checkpoint)
  if chk_path.exists():
    return chk_path

  repo_dir = epath.Path(ligandmpnn_dir)
  candidate = repo_dir / 'model_params' / checkpoint
  if candidate.exists():
    return candidate

  raise FileNotFoundError(
      f'LigandMPNN checkpoint "{checkpoint}" not found at {chk_path} or'
      f' {candidate}. Make sure you have downloaded weights using `bash'
      ' get_model_params.sh ./model_params` inside your LigandMPNN directory.'
  )


PDB_PATH_MULTI_FILENAME = 'pdb_path_multi.json'
FIXED_RESIDUES_MULTI_FILENAME = 'fixed_residues_multi.json'


def write_multi_input_files(
    designs: Sequence[DesignInputs],
    output_dir: epath.PathLike,
    *,
    pdb_filename: str = PDB_PATH_MULTI_FILENAME,
    fixed_filename: str = FIXED_RESIDUES_MULTI_FILENAME,
) -> tuple[epath.Path, epath.Path]:
  """Writes the `--pdb_path_multi` and `--fixed_residues_multi` JSON files.

  Args:
    designs: Sequence of `DesignInputs` to resequence in a single batch call.
    output_dir: Directory where the multi-input JSON files are written.
    pdb_filename: Filename for the `--pdb_path_multi` JSON file.
    fixed_filename: Filename for the `--fixed_residues_multi` JSON file.

  Returns:
    Tuple of `(pdb_path_multi_path, fixed_residues_multi_path)`.
  """
  out_dir = epath.Path(output_dir)
  pdb_multi = {str(design.pdb_path): '' for design in designs}
  fixed_multi = {
      str(design.pdb_path): ' '.join((design.fixed_residues or '').split())
      for design in designs
  }
  pdb_multi_path = out_dir / pdb_filename
  fixed_multi_path = out_dir / fixed_filename
  pdb_multi_path.write_text(json.dumps(pdb_multi, indent=2) + '\n')
  fixed_multi_path.write_text(json.dumps(fixed_multi, indent=2) + '\n')
  return pdb_multi_path, fixed_multi_path


def build_ligandmpnn_command(
    python_executable: str,
    ligandmpnn_dir: epath.PathLike,
    pdb_path_multi: epath.PathLike,
    fixed_residues_multi: epath.PathLike,
    output_dir: epath.PathLike,
    checkpoint_path: epath.PathLike,
    model_type: str,
    temperature: float,
    use_atom_context: int,
    cutoff_for_score: float,
    use_side_chain_context: int,
    omit_aa: str,
    num_sequences: int = 1,
) -> list[str]:
  """Constructs the command-line argument list for LigandMPNN run.py."""
  if num_sequences < 1:
    raise ValueError(f'num_sequences must be >= 1 (got {num_sequences}).')
  run_script = epath.Path(ligandmpnn_dir) / 'run.py'
  if not run_script.exists():
    raise FileNotFoundError(
        f'LigandMPNN run script not found at: {run_script}. Please check'
        ' --ligandmpnn_dir.'
    )

  # LigandMPNN has a bug when batch_size > 1, so we always set --batch_size=1
  # and pass num_sequences via --number_of_batches.
  return [
      python_executable,
      str(run_script),
      *('--model_type', str(model_type)),
      *('--checkpoint_ligand_mpnn', str(checkpoint_path)),
      *('--pdb_path_multi', str(pdb_path_multi)),
      *('--fixed_residues_multi', str(fixed_residues_multi)),
      *('--out_folder', str(output_dir)),
      *('--temperature', str(temperature)),
      *('--ligand_mpnn_use_atom_context', str(use_atom_context)),
      *('--ligand_mpnn_cutoff_for_score', str(cutoff_for_score)),
      *('--ligand_mpnn_use_side_chain_context', str(use_side_chain_context)),
      *('--omit_AA', str(omit_aa)),
      *('--batch_size', '1'),
      *('--number_of_batches', str(num_sequences)),
  ]


def parse_fasta_output(fasta_path: epath.PathLike) -> list[tuple[str, str]]:
  """Parses a FASTA file into a list of (header, sequence) pairs."""
  path = epath.Path(fasta_path)
  if not path.exists():
    raise FileNotFoundError(f'FASTA file not found at: {path}')

  lines = [
      line.strip() for line in path.read_text().splitlines() if line.strip()
  ]
  entries = []
  current_header = None
  current_seq: list[str] = []

  for line in lines:
    if line.startswith('>'):
      if current_header is not None:
        entries.append((current_header, ''.join(current_seq)))
        current_seq = []
      current_header = line
    else:
      current_seq.append(line)

  if current_header is not None:
    entries.append((current_header, ''.join(current_seq)))

  return entries


def extract_redesigned_sequences(
    fasta_path: epath.PathLike,
    num_sequences: int = 1,
) -> list[tuple[str, str]]:
  """Extracts the redesigned sequences from LigandMPNN FASTA output.

  Args:
    fasta_path: Path to the FASTA file generated by LigandMPNN.
    num_sequences: Expected number of redesigned sequences in the FASTA file
      (excluding the original backbone entry at index 0).

  Returns:
    A list of `(redesigned_header, redesigned_sequence)` pairs of length
    `num_sequences`.

  Raises:
    FileNotFoundError: If the FASTA file does not exist.
    ValueError: If the FASTA file does not contain `1 + num_sequences` entries.
  """
  path = epath.Path(fasta_path)
  if not path.exists():
    raise FileNotFoundError(
        f'Expected LigandMPNN output FASTA not found at: {path}'
    )

  logging.info('Redesign complete! Generated sequences at: %s', path)
  entries = parse_fasta_output(path)
  for i, (hdr, seq) in enumerate(entries):
    logging.info('Entry %d: %s | length=%d', i, hdr, len(seq))

  expected_entries = 1 + num_sequences
  if len(entries) != expected_entries:
    raise ValueError(
        f'Expected {expected_entries} entries in FASTA {path} (original'
        f' backbone and {num_sequences} redesigned sequence(s)), but got'
        f' {len(entries)}'
    )

  return entries[1:]


def extract_single_redesigned_sequence(
    fasta_path: epath.PathLike,
) -> tuple[str, str]:
  """Extracts and saves the single redesigned sequence from LigandMPNN output."""
  path = epath.Path(fasta_path)
  redesigned_hdr, redesigned_seq = extract_redesigned_sequences(
      path, num_sequences=1
  )[0]
  path.write_text(f'{redesigned_hdr}\n{redesigned_seq}\n')
  logging.info('Saved single redesigned sequence to %s', path)
  return redesigned_hdr, redesigned_seq


def _is_design_resequenced(
    prefix: str,
    num_sequences: int,
    existing_out_names: frozenset[str],
) -> bool:
  """Returns True when all expected resequenced mmCIFs exist for `prefix`."""
  return all(
      (
          f'{design_manifest.resequenced_prefix(prefix, reseq_index=idx, num_sequences=num_sequences)}'
          f'{design_manifest.RESEQ_CIF_SUFFIX}'
      )
      in existing_out_names
      for idx in range(num_sequences)
  )


def _finalize_redesigned_design(
    *,
    resolved_pdb: epath.Path,
    resolved_cif: epath.Path,
    resolved_fixed_residues: str | None,
    out_dir: epath.Path,
    num_sequences: int = 1,
    stale_files: Sequence[epath.Path] = (),
) -> None:
  """Extracts redesigned FASTA(s) and writes resequenced mmCIF(s)."""
  prefix = resolved_pdb.stem
  fasta_filename = f'{prefix}{design_manifest.FASTA_SUFFIX}'
  expected_fa = out_dir / 'seqs' / fasta_filename
  if num_sequences == 1:
    _, single_seq = extract_single_redesigned_sequence(expected_fa)
    redesigned_entries = [('', single_seq)]
  else:
    redesigned_entries = extract_redesigned_sequences(
        expected_fa, num_sequences=num_sequences
    )

  for stale in stale_files:
    stale.unlink(missing_ok=True)

  logging.info('Creating resequenced structure(s) from %s', resolved_cif)
  base_struct = structure.from_mmcif(resolved_cif.read_text())

  for idx, (_, redesigned_seq) in enumerate(redesigned_entries):
    reseq_prefix = design_manifest.resequenced_prefix(
        prefix, reseq_index=idx, num_sequences=num_sequences
    )
    fasta_path = out_dir / f'{reseq_prefix}{design_manifest.FASTA_SUFFIX}'
    fasta_path.write_text(
        structure_utils.format_fasta_with_terms_of_use(
            reseq_prefix, redesigned_seq
        )
    )
    logging.info('Saved redesigned FASTA to %s', fasta_path)

    reseq_struct = structure_utils.create_resequenced_structure(
        base_struct=base_struct,
        redesigned_seq=redesigned_seq,
        fixed_residues=resolved_fixed_residues,
    )
    reseq_cif = out_dir / f'{reseq_prefix}{design_manifest.RESEQ_CIF_SUFFIX}'
    reseq_cif.write_text(
        structure_utils.add_license_and_terms_of_use_header(
            reseq_struct.to_mmcif()
        )
    )
    logging.info('Saved resequenced mmCIF structure to %s', reseq_cif)


def _finalize_worker(
    task: tuple[
        epath.Path,
        epath.Path,
        str | None,
        epath.Path,
        int,
        tuple[epath.Path, ...],
    ],
) -> None:
  """Worker wrapper for parallelizing `_finalize_redesigned_design`."""
  (
      resolved_pdb,
      resolved_cif,
      resolved_fixed_residues,
      out_dir,
      num_sequences,
      stale_files,
  ) = task
  _finalize_redesigned_design(
      resolved_pdb=resolved_pdb,
      resolved_cif=resolved_cif,
      resolved_fixed_residues=resolved_fixed_residues,
      out_dir=out_dir,
      num_sequences=num_sequences,
      stale_files=stale_files,
  )


def run_ligandmpnn_redesign(
    input_dir: epath.PathLike | None,
    output_dir: epath.PathLike,
    ligandmpnn_dir: epath.PathLike,
    checkpoint: str,
    model_type: str,
    temperature: float,
    fixed_residues_override: str | None,
    use_atom_context: int,
    cutoff_for_score: float,
    use_side_chain_context: int,
    omit_aa: str,
    num_sequences: int,
    python_executable: str,
    resume: bool = False,
    num_workers: int = 1,
) -> epath.Path:
  """Redesigns the sequence of every design in `input_dir`."""
  if num_sequences < 1:
    raise ValueError(f'num_sequences must be >= 1 (got {num_sequences}).')
  out_dir = epath.Path(output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  existing_out_names = design_manifest.list_dir_names(out_dir)

  designs = _resolve_designs(
      input_dir=input_dir,
      fixed_residues_override=fixed_residues_override,
  )

  chk_path = resolve_checkpoint(ligandmpnn_dir, checkpoint)
  logging.info('Using LigandMPNN checkpoint: %s', chk_path)
  logging.info('Resequencing %d design(s).', len(designs))

  pending = []
  skipped = 0
  for design in designs:
    if resume and _is_design_resequenced(
        design.pdb_path.stem,
        num_sequences,
        existing_out_names=existing_out_names,
    ):
      logging.info('%s is already resequenced.', design.pdb_path.stem)
      skipped += 1
      continue
    pending.append(design)

  if pending:
    pdb_multi_path, fixed_multi_path = write_multi_input_files(pending, out_dir)
    workers = max(1, min(num_workers, len(pending)))
    if workers > 1:
      (out_dir / 'seqs').mkdir(parents=True, exist_ok=True)
      (out_dir / 'backbones').mkdir(parents=True, exist_ok=True)
      logging.info(
          'Sharding %d pending design(s) across %d LigandMPNN CPU worker(s).',
          len(pending),
          workers,
      )
      worker_env = {
          **os.environ,
          'CUDA_VISIBLE_DEVICES': '',
          'OMP_NUM_THREADS': '1',
          'MKL_NUM_THREADS': '1',
      }
      procs = []
      shard_files = []
      for w in range(workers):
        shard_pdb, shard_fixed = write_multi_input_files(
            pending[w::workers],
            out_dir,
            pdb_filename=f'pdb_path_multi_shard{w}.json',
            fixed_filename=f'fixed_residues_multi_shard{w}.json',
        )
        shard_files.extend([shard_pdb, shard_fixed])
        cmd = build_ligandmpnn_command(
            python_executable=python_executable,
            ligandmpnn_dir=ligandmpnn_dir,
            pdb_path_multi=shard_pdb,
            fixed_residues_multi=shard_fixed,
            output_dir=out_dir,
            checkpoint_path=chk_path,
            model_type=model_type,
            temperature=temperature,
            use_atom_context=use_atom_context,
            cutoff_for_score=cutoff_for_score,
            use_side_chain_context=use_side_chain_context,
            omit_aa=omit_aa,
            num_sequences=num_sequences,
        )
        procs.append(subprocess.Popen(cmd, env=worker_env))
      exit_codes = [p.wait() for p in procs]
      for sf in shard_files:
        sf.unlink(missing_ok=True)
      if any(rc != 0 for rc in exit_codes):
        raise subprocess.CalledProcessError(
            next(rc for rc in exit_codes if rc != 0),
            'LigandMPNN CPU worker',
        )
    else:
      cmd = build_ligandmpnn_command(
          python_executable=python_executable,
          ligandmpnn_dir=ligandmpnn_dir,
          pdb_path_multi=pdb_multi_path,
          fixed_residues_multi=fixed_multi_path,
          output_dir=out_dir,
          checkpoint_path=chk_path,
          model_type=model_type,
          temperature=temperature,
          use_atom_context=use_atom_context,
          cutoff_for_score=cutoff_for_score,
          use_side_chain_context=use_side_chain_context,
          omit_aa=omit_aa,
          num_sequences=num_sequences,
      )
      logging.info('Executing command:\n%s', ' '.join(cmd))
      prev_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
      os.environ['CUDA_VISIBLE_DEVICES'] = ''
      try:
        subprocess.run(cmd, check=True)
      finally:
        if prev_cuda_visible_devices is None:
          os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
          os.environ['CUDA_VISIBLE_DEVICES'] = prev_cuda_visible_devices

    # Pre-group any existing `<prefix>_seq*` files from prior multi-sequence
    # runs so each design's finalization can delete stale outputs without
    # repeating full-directory glob scans.
    stale_by_prefix: dict[str, list[epath.Path]] = {}
    for fname in existing_out_names:
      prefix, sep, _ = fname.rpartition('_seq')
      if sep and prefix:
        stale_by_prefix.setdefault(prefix, []).append(out_dir / fname)

    if workers > 1:
      tasks = [
          (
              d.pdb_path,
              d.cif_path,
              d.fixed_residues,
              out_dir,
              num_sequences,
              tuple(stale_by_prefix.get(d.pdb_path.stem, ())),
          )
          for d in pending
      ]
      with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_finalize_worker, tasks))
    else:
      for design in pending:
        _finalize_redesigned_design(
            resolved_pdb=design.pdb_path,
            resolved_cif=design.cif_path,
            resolved_fixed_residues=design.fixed_residues,
            out_dir=out_dir,
            num_sequences=num_sequences,
            stale_files=stale_by_prefix.get(design.pdb_path.stem, ()),
        )

  logging.info(
      'Resequencing completed: %d design(s) redesigned, %d skipped.',
      len(pending),
      skipped,
  )
  return out_dir


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many positional arguments.')

  logging.info(
      'AlphaProtein Novo LigandMPNN Sequence Redesign runner starting.'
  )
  run_ligandmpnn_redesign(
      input_dir=_INPUT_DIR.value,
      output_dir=_OUTPUT_DIR.value,
      ligandmpnn_dir=_LIGANDMPNN_DIR.value,
      checkpoint=_CHECKPOINT.value,
      model_type=_MODEL_TYPE.value,
      temperature=_TEMPERATURE.value,
      fixed_residues_override=_FIXED_RESIDUES.value,
      use_atom_context=_USE_ATOM_CONTEXT.value,
      cutoff_for_score=_CUTOFF_FOR_SCORE.value,
      use_side_chain_context=_USE_SIDE_CHAIN_CONTEXT.value,
      omit_aa=_OMIT_AA.value,
      num_sequences=_NUM_SEQUENCES.value,
      python_executable=_PYTHON_EXECUTABLE.value,
      resume=_RESUME.value,
      num_workers=_NUM_WORKERS.value,
  )
  logging.info('LigandMPNN runner finished successfully.')


if __name__ == '__main__':
  app.run(main)
