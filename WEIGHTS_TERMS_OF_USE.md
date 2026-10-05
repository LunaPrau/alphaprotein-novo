# ALPHAPROTEIN NOVO GENERATOR MODEL PARAMETERS TERMS OF USE

**Last Modified:** 2026-10-05

**AlphaProtein Novo** is a generative design pipeline for *de novo* protein and
enzyme design developed by [Google DeepMind](https://deepmind.google/). The
pipeline consists of the AP Novo Generator (as defined below), followed by
optional resequencing with LigandMPNN, AlphaFold 3 structure prediction and
filtering based on the metrics from AlphaFold 3 (collectively the “**AP Novo
Pipeline**”). LigandMPNN and AlphaFold 3 are governed by their respective terms,
conditions and licenses.

The AP Novo Generator generates 3D protein structures and amino acid sequences
conditioned on Input (as defined below). We make the trained AP Novo Generator
model parameters and output generated using those available free of charge for
certain non-commercial uses in accordance with these terms of use and the
[AP Novo Generator Model Parameters Prohibited Use Policy](https://github.com/google-deepmind/alphaprotein-novo/blob/main/WEIGHTS_PROHIBITED_USE_POLICY.md).

## Key things to know when using the AP Novo Generator Model Parameters and Output

-   The Model Parameters and Output (each as defined below) are only available
    for non-commercial use by, or on behalf of, non-commercial organizations
    (i.e., universities, non-profit organizations and research institutes,
    educational, journalism and government bodies). If you are a researcher
    affiliated with a non-commercial organization, provided you are not a
    commercial organization or acting on behalf of a commercial organization,
    this means you can use these for your non-commercial affiliated research.

-   You must not use nor allow others to use:

    -   **the Model Parameters or Output in connection with any commercial
        activities, including research on behalf of commercial organizations;
        or**

    -   **Output to train, fine-tune, or distill machine learning models or
        related technology for** ***de novo*** **protein or enzyme design,
        biomolecular structure generation, or protein sequence design similar to
        the AP Novo Generator.**

-   You must not publish or share the Model Parameters, except sharing these
    within your organization in accordance with these Terms.

-   You can publish, share and adapt Output in accordance with these Terms,
    including the requirements to provide clear notice of any modifications you
    make and that ongoing use of Output and derivatives are subject to the
    [AP Novo Generator Output Terms of Use](https://github.com/google-deepmind/alphaprotein-novo/blob/main/OUTPUT_TERMS_OF_USE.md).

By using, reproducing, modifying, performing, distributing or displaying any
portion or element of the Model Parameters (as defined below) or otherwise
accepting the terms of this agreement, you agree to be bound by (1) these terms
of use, and (2) the
[AP Novo Generator Model Parameters Prohibited Use Policy](https://github.com/google-deepmind/alphaprotein-novo/blob/main/WEIGHTS_PROHIBITED_USE_POLICY.md)
which is incorporated herein by reference (together, the **"Terms"**), in each
case (a) as modified from time to time in accordance with the Terms, and (b)
between you and (i) if you are from a country in the European Economic Area or
Switzerland, Google Ireland Limited, or (ii) otherwise, Google LLC.

You confirm you are authorized either explicitly or implicitly to enter, and are
entering, into the Terms as an employee on behalf of, or otherwise on behalf of,
your organization.

Please read these Terms carefully. They establish what you can expect from us as
you access and use the AP Novo Generator Assets (as defined below), and what
Google expects from you. When we say **"you"**, we mean the individual or
organization using the AP Novo Generator Assets. When we say **"we"**, **"us"**
or **"Google"**, we mean the entities that belong to the Google group of
companies, which means Google LLC and its affiliates.

--------------------------------------------------------------------------------

## 1. Key Definitions

As used in these Terms:

-   “**AP Novo Generator**" means the generative diffusion model for *de novo*
    protein structure and sequence generation, consisting of the (a) AP Novo
    Generator Code, and (b) the Model Parameters.

-   "**AP Novo Generator Assets**" means the Model Parameters and Output.

-   **"AP Novo Generator Code"** means the source code for the AP Novo Generator
    made available at
    [https://github.com/google-deepmind/alphaprotein-novo](https://github.com/google-deepmind/alphaprotein-novo)
    and licensed under the terms of the Apache 2.0 license and any derivative
    source code.

-   **“AP Novo Pipeline”** has the meaning given above.

-   **“Derived Output”** means output from any model or tool that forms part of
    the AP Novo Pipeline (for example, LigandMPNN) to the extent the input used
    to generate that output includes output from the AP Novo Generator.

-   **"Distribution"** or **"Distribute"** means any transmission, publication,
    transfer, or other sharing of Output (whether in digital, sequence, or
    physical biological/chemical form) publicly or to any other person.

-   **"Input"** means any data, biomolecular structures, atomic coordinates,
    catalytic motif specifications, residue constraints, ligand or substrate
    coordinates or chemical identifiers (such as CCD codes), or other
    conditioning information provided to or used with the AP Novo Generator.

-   "**Model Parameters**" means the trained AP Novo Generator model weights and
    parameters made available by Google (at its sole discretion) for their use
    in accordance with these Terms, together with (a) modifications to those
    weights and parameters, (b) works based on those weights and parameters, or
    (c) other code or machine learning models which incorporate, in full or in
    part, those weights and parameters.

-   **"Output"** means the generated 3D biomolecular structures, protein
    backbones, atomic coordinates, and initial amino acid sequences provided by
    the AP Novo Generator or using the Model Parameters, together with all
    ancillary and related generation metadata provided by the AP Novo
    Generator**,** as well as visual representations, descriptions,
    modifications, copies, or adaptations that are substantially derived from
    such output, including all Derived Output. For the avoidance of doubt,
    Output does not include your pre-existing Input in isolation, though any
    composite structure or file emitted by the AP Novo Generator that
    incorporates Input motif or ligand coordinates is governed as Output when
    Distributed as part of the generated design.

-   **"Including"** means "**including without limitation**".

--------------------------------------------------------------------------------

## 2. Accessing and using the AP Novo Generator Assets

Subject to your compliance with the Terms, including the
[AP Novo Generator Model Parameters Prohibited Use Policy](https://github.com/google-deepmind/alphaprotein-novo/blob/main/WEIGHTS_PROHIBITED_USE_POLICY.md),
you may access, use and modify the AP Novo Generator Assets and Distribute the
Output as set out in these Terms. We grant you a non-exclusive, royalty-free,
revocable, non-transferable and non-sublicensable (except as expressly permitted
in these Terms) license to any intellectual property rights we have in the AP
Novo Generator Assets to the extent necessary for these purposes. In order to
verify your access and use of AP Novo Generator Assets, we may from time-to-time
ask for additional information from you, including verification of your name,
organization, and other identifying information.

By accessing, using, or modifying the AP Novo Generator Assets, Distributing
Output, or requesting to access the Model Parameters, you represent and warrant
that:

1.  you have full power and authority to enter into these Terms (including being
    of sufficient age of consent),

2.  Google has never previously terminated your access and right to use the AP
    Novo Generator, or AlphaFold 3 (including as made available via
    [AlphaFold Server](https://alphafoldserver.com/about)) due to your breach of
    applicable terms of use,

3.  entering into or performing your rights and obligations under these Terms
    (including providing and using any Input with the AP Novo Generator) will
    not violate any agreement you have with a third party or any third-party
    rights, and you have all necessary rights, licenses, and permissions to
    provide and use all Input with the AP Novo Generator,

4.  any information provided by you to Google in relation to the AP Novo
    Pipeline, including (where applicable) in order to request access to the
    Model Parameters, is correct and current, and

5.  you are not (i) resident of an embargoed country, (ii) ordinarily resident
    in a US embargoed country, or (iii) otherwise prohibited by applicable
    export controls and sanctions programs from accessing, using, or modifying
    the AP Novo Generator Assets.

If you choose to give Google feedback, such as suggestions to improve the AP
Novo Generator or AP Novo Pipeline, you undertake any such information is
non-confidential and non-proprietary, and Google may act on your feedback
without obligation to you.

--------------------------------------------------------------------------------

## 3. Use Restrictions

You must not use any of the AP Novo Generator Assets, nor provide or process any
Input with the AP Novo Generator:

-   for the restricted uses set forth in the
    [AP Novo Generator Model Parameters Prohibited Use Policy](https://github.com/google-deepmind/alphaprotein-novo/blob/main/WEIGHTS_PROHIBITED_USE_POLICY.md);
    or

-   in violation of applicable laws and regulations (including applicable
    biosafety, biosecurity, gene synthesis screening, and export or chemical
    control laws).

To the maximum extent permitted by law and without limiting any of our other
rights, Google reserves the right to revoke your right to use, and (to the
extent feasible) restrict usage of any of the AP Novo Generator Assets that
Google reasonably believes is in violation of these Terms.

--------------------------------------------------------------------------------

## 4. Input and Generated Output

Although you must comply with these Terms when providing Input to the AP Novo
Generator and using the AP Novo Generator Assets, we do not claim ownership in
the Input you provide to the AP Novo Generator or in the original Output you
generate using the AP Novo Generator. To the extent that any Output incorporates
portions of your Input (such as fixed input catalytic motif coordinates or
ligand structures preserved in generated structure files), your use and
Distribution of that composite Output is subject to these Terms and the AP Novo
Generator Output Terms of Use, though your standalone, pre-existing Input
remains yours. You acknowledge that the AP Novo Generator may generate the same
or similar Output for multiple users, including Google, and we reserve all our
rights in this respect.

--------------------------------------------------------------------------------

## 5. Changes to the AP Novo Generator Assets or these Terms

Google may add or remove functionalities or features of the AP Novo Generator
Assets at any time and may stop offering access to the AP Novo Generator Assets
altogether.

Google may update these Terms and the access mechanism for the Model Parameters
at any time. We will post any modifications to the Terms
[in the AP Novo GitHub repository](https://github.com/google-deepmind/alphaprotein-novo).
Changes will generally become effective 14 days after they are posted. However,
changes addressing functionality or made for legal or biosecurity reasons will
be effective immediately.

You should review the Terms whenever we update them or you use the AP Novo
Generator Assets. If you do not agree to any modifications to the Terms, you
must stop using the AP Novo Generator Assets immediately.

--------------------------------------------------------------------------------

## 6. Suspending or terminating your right to use the AP Novo Generator Assets

Google may at any time suspend or terminate your right to use and, as applicable
access to, the AP Novo Generator Assets because of, among other reasons, your
failure to fully comply with the Terms. If Google suspends or terminates your
right to access or use the AP Novo Generator Assets, you must immediately delete
and cease use and Distribution of all copies of the AP Novo Generator Assets in
your possession or control and are prohibited from using the AP Novo Generator
Assets, including by submitting an application to use the Model Parameters.
Google will endeavour to give you reasonable notice prior to any such suspension
or termination, but no notice or prior warning will be given if the suspension
or termination is for your failure to fully comply with the Terms or other
serious grounds. Of course, you are always free to stop using the AP Novo
Generator Assets.

--------------------------------------------------------------------------------

## 7. Confidentiality

You agree not to disclose or make available Google Confidential Information to
anyone without our prior written consent. "Google Confidential Information"
means all software, information, technology and documentation relating to the AP
Novo Generator or AP Novo Pipeline made available by Google that is marked
confidential or would normally be considered confidential under the
circumstances in which it is presented. Google Confidential Information does not
include (a) information that you already knew prior to your access to, or use
of, the AP Novo Generator Assets, (b) that becomes public through no fault of
yours (for example, your breach of the Terms), (c) that was independently
developed by you without reference to Google Confidential Information, or (d)
that was lawfully given to you by a third party (without your or their breach of
the Terms).

--------------------------------------------------------------------------------

## 8. Disclaimers

Nothing in the Terms restricts any rights that cannot be restricted under
applicable law or limits Google's responsibilities except as allowed by
applicable law.

AP Novo Generator Assets are provided on an "as is" basis, without warranties or
conditions of any kind, either express or implied, including any warranties or
conditions of title, non-infringement, merchantability, catalytic activity,
stability, selectivity or fitness for a particular purpose. You are solely
responsible for determining the appropriateness of using the Model Parameters,
or using, synthesizing, expressing, or Distributing Output, and assume any and
all risks associated with such use or distribution and your exercise of rights
and obligations under these Terms. You and anyone you share Output with are
solely responsible for these and their subsequent uses.

Output are computational designs and predictions with varying levels of
confidence and should be interpreted carefully. Use discretion before relying
on, publishing, downloading, synthesizing, or otherwise using Model Parameters.

**Experimental Synthesis, Wet-Lab Expression, and Biocatalysis:** Outputs are
computational designs whose physical, chemical, toxicologic, and biological
properties have not been experimentally verified or safety-certified by Google.
If you choose to physically synthesize nucleic acids, express proteins or
enzymes, or perform chemical/biocatalytic reactions derived from Output in a
laboratory setting, you do so solely at your own risk. You are solely
responsible for: (a) conducting appropriate biosecurity and gene synthesis
screening, (b) ensuring compliance with all applicable institutional biosafety
(IBC), chemical safety, environmental, and regulatory standards, and (c) safely
handling any reagents, substrates, catalytic intermediates, or reaction
products.

AP Novo Generator Assets are for non-commercial scientific research only. The AP
Novo Generator and the AP Novo Generator Assets are not intended, validated, or
approved for direct clinical, diagnostic, or human therapeutic administration.
You should not use them for clinical purposes or rely on them for medical or
other professional advice. Any content regarding those topics is provided for
informational purposes only and is not a substitute for advice from a qualified
professional.

--------------------------------------------------------------------------------

## 9. Liabilities

To the extent allowed by applicable law, you will indemnify Google and its
directors, officers, employees, and contractors for any third-party legal
proceedings (including actions by government authorities) arising out of or
relating to your unlawful use of the AP Novo Generator Assets, your provision or
use of any Input, any physical synthesis or wet-lab use of Output, or violation
of the Terms. This indemnity covers any liability or expense arising from
claims, losses, damages, judgments, fines, litigation costs, and legal fees,
except to the extent a liability or expense is caused by Google's breach,
negligence, or willful misconduct. If you are legally exempt from certain
responsibilities, including indemnification, then those responsibilities do not
apply to you under the Terms.

In no circumstances will Google be responsible for any indirect, special,
incidental, exemplary, consequential, or punitive damages, or lost profits of
any kind in connection with the Terms or the AP Novo Generator Assets, even if
Google has been advised of the possibility of such damages. Google's total
aggregate liability for all claims arising out of or in connection with the
Terms or the AP Novo Generator Assets, including for its own negligence, is
limited to $500.

--------------------------------------------------------------------------------

## 10. Miscellaneous

By law, you have certain rights that cannot be limited by a contract like the
Terms. The Terms are in no way intended to restrict those rights.

The Terms are our entire agreement relating to your use of the AP Novo Generator
Assets and supersede any prior or contemporaneous agreements on that subject.

If it turns out that a particular provision of the Terms is not enforceable, the
balance of the Terms will remain in full force and effect.

--------------------------------------------------------------------------------

## 11. Disputes

California law will govern all disputes arising out of or relating to the Terms
or in connection to the AP Novo Generator Assets. These disputes will be
resolved exclusively in the federal or state courts of Santa Clara County,
California, USA and you and Google consent to personal jurisdiction in those
courts. To the extent that applicable local law prevents certain disputes from
being resolved in a California court, you and Google can file those disputes in
your local courts. If applicable local law prevents your local court from
applying California law to resolve these disputes, then these disputes will be
governed by the applicable local laws of your country, state, or other place of
residence. If you are using the AP Novo Generator Assets on behalf of a
government organization other than US federal government organizations (where
the foregoing provisions shall apply to the extent permitted by federal law),
these Terms will be silent regarding governing law and courts.

Given the nature of scientific research, it may take some time for any breach of
the Terms to become apparent. To protect you, Google and the AP Novo Generator
Assets, to the extent allowed by applicable law you agree that:

-   any legal claims relating to the Terms or the AP Novo Generator Assets can
    be initiated until the later of:

    -   the cut-off date under applicable law for bringing the legal claim; or

    -   two years from the date you or Google (as applicable) became aware, or
        should reasonably have become aware, of the facts giving rise to that
        claim.

-   you will not argue limitation, time bar, delay, waiver, or the like in an
    attempt to bar an action filed within that time period, and neither will
    Google.

All rights not specifically and expressly granted to you by the Terms are
reserved to Google. No delay, act or omission by Google in exercising any right
or remedy will be deemed a waiver of any breach of the Terms and Google
expressly reserves any and all rights and remedies available under the Terms or
at law or in equity or otherwise, including the remedy of injunctive relief
against any threatened or actual breach of the Terms without the necessity of
proving actual damages.
