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

"""Links to the AlphaProtein Novo repository and its terms of use."""

from typing import Final

GITHUB_REPO_URL: Final[str] = (
    'https://github.com/google-deepmind/alphaprotein-novo'
)
OUTPUT_TERMS_OF_USE_URL: Final[str] = (
    f'{GITHUB_REPO_URL}/blob/main/OUTPUT_TERMS_OF_USE.md'
)
WEIGHTS_TERMS_OF_USE_URL: Final[str] = (
    f'{GITHUB_REPO_URL}/blob/main/WEIGHTS_TERMS_OF_USE.md'
)

# The notice that the AP Novo Generator Output Terms of Use require to
# accompany any Distribution of Output, reproduced verbatim and split into
# lines. Both of the notice renderings below are derived from this single
# tuple so that their wording cannot drift apart.
_OUTPUT_NOTICE_LINES: Final[tuple[str, ...]] = (
    'By using this information, structure, or sequence, you agree to the AP',
    'Novo Generator Output Terms of Use found at',
    f'{OUTPUT_TERMS_OF_USE_URL}.',
    'To request access to the AP Novo Generator model parameters, follow the',
    f'process set out at {GITHUB_REPO_URL}. You',
    'may only use these if received directly from Google. Use is subject to',
    'terms of use available at',
    f'{WEIGHTS_TERMS_OF_USE_URL}.',
)

# The notice as a `#` comment block, prepended to mmCIF and PDB output. `#` is
# the comment character in the CIF/STAR grammar, and PDB readers skip records
# with unrecognised leading keywords, so both formats tolerate the block.
#
# This rendering must NOT be used for FASTA. Most FASTA readers (Biopython's
# default `fasta` parser, biotite, pyfastx, and htslib's `faidx` indexer)
# reject a file that does not begin with `>`, so a leading comment block makes
# the file unreadable. FASTA output uses `OUTPUT_FASTA_HEADER_NOTICE` instead.
OUTPUT_FILE_NOTICE: Final[str] = '\n'.join(
    f'# {line}' for line in _OUTPUT_NOTICE_LINES
)

# The notice as a single line, carried in the description field of a FASTA
# header. The description is free text that follows the record ID, so readers
# parse the record normally and the ID remains the text up to the first space.
OUTPUT_FASTA_HEADER_NOTICE: Final[str] = ' '.join(_OUTPUT_NOTICE_LINES)
