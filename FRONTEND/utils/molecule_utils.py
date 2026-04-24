"""Molecule visualization utilities using RDKit."""
import io
import base64
from typing import Optional

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem import rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def smiles_to_image_b64(smiles: str, size: tuple = (300, 200)) -> Optional[str]:
    """Convert SMILES string to base64-encoded PNG image."""
    if not RDKIT_AVAILABLE or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=size, kekulize=True)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


def smiles_to_svg(smiles: str, size: tuple = (300, 200)) -> Optional[str]:
    """Convert SMILES string to SVG string."""
    if not RDKIT_AVAILABLE or not smiles:
        return None
    try:
        from rdkit.Chem.Draw import rdMolDraw2D
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return None


def get_mol_properties(smiles: str) -> dict:
    """Extract basic molecular properties."""
    if not RDKIT_AVAILABLE or not smiles:
        return {}
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        return {
            "num_atoms":  mol.GetNumAtoms(),
            "num_bonds":  mol.GetNumBonds(),
            "mol_weight": round(rdMolDescriptors.CalcExactMolWt(mol), 2),
            "num_rings":  rdMolDescriptors.CalcNumRings(mol),
            "num_hba":    rdMolDescriptors.CalcNumHBA(mol),
            "num_hbd":    rdMolDescriptors.CalcNumHBD(mol),
        }
    except Exception:
        return {}
