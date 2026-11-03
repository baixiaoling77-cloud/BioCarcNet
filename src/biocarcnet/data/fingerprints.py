"""Fingerprint features used by BioCarcNet."""

from skfp.fingerprints import RDKitFingerprint


def get_RDKitFingerprint(smiles):
    return RDKitFingerprint().transform(smiles)
