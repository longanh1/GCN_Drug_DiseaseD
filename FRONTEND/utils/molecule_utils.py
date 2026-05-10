"""Molecule visualization utilities using RDKit."""
import io
import base64
from typing import Optional

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
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


def smiles_to_3d_plotly(smiles: str, title: str = ""):
    """Generate interactive 3D plotly figure from SMILES using RDKit 3D coordinates."""
    if not RDKIT_AVAILABLE or not smiles:
        return None
    try:
        import plotly.graph_objects as go

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mol_h = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
        if result != 0:
            result = AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())
        if result != 0:
            # fallback: random coords
            result = AllChem.EmbedMolecule(mol_h, randomSeed=42)
        if result != 0:
            return None

        try:
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
        except Exception:
            pass

        conf = mol_h.GetConformer()

        ELEMENT_COLORS = {
            'C': '#555555', 'H': '#cccccc', 'O': '#ff4444',
            'N': '#4466ff', 'S': '#ddcc00', 'F': '#44ddcc',
            'Cl': '#44dd44', 'Br': '#aa2200', 'P': '#ff8800',
            'I': '#993399', 'Si': '#999900', 'B': '#ff8888',
        }
        ELEMENT_SIZES = {
            'C': 10, 'H': 5, 'O': 12, 'N': 11, 'S': 14,
            'F': 9, 'Cl': 13, 'Br': 14, 'P': 13, 'I': 15,
        }

        atoms_x, atoms_y, atoms_z = [], [], []
        atom_colors, atom_sizes, atom_labels = [], [], []

        for atom in mol_h.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            sym = atom.GetSymbol()
            atoms_x.append(pos.x)
            atoms_y.append(pos.y)
            atoms_z.append(pos.z)
            atom_colors.append(ELEMENT_COLORS.get(sym, '#888888'))
            atom_sizes.append(ELEMENT_SIZES.get(sym, 8))
            atom_labels.append(sym if sym != 'H' else '')

        bond_x, bond_y, bond_z = [], [], []
        for bond in mol_h.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            pi = conf.GetAtomPosition(i)
            pj = conf.GetAtomPosition(j)
            bond_x += [pi.x, pj.x, None]
            bond_y += [pi.y, pj.y, None]
            bond_z += [pi.z, pj.z, None]

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=bond_x, y=bond_y, z=bond_z,
            mode='lines',
            line=dict(color='rgba(160,160,160,0.6)', width=4),
            hoverinfo='none',
            showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=atoms_x, y=atoms_y, z=atoms_z,
            mode='markers+text',
            marker=dict(
                color=atom_colors,
                size=atom_sizes,
                opacity=0.92,
                line=dict(width=0.8, color='rgba(255,255,255,0.4)'),
            ),
            text=atom_labels,
            textfont=dict(size=8, color='#ffffff'),
            hovertemplate='<b>%{text}</b><br>x=%{x:.2f} Å<br>y=%{y:.2f} Å<br>z=%{z:.2f} Å<extra></extra>',
            showlegend=False,
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=11, color='#818cf8')),
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                           showbackground=False, title=''),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                           showbackground=False, title=''),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                           showbackground=False, title=''),
                bgcolor='rgba(15,20,40,0.95)',
            ),
            margin=dict(t=30, b=0, l=0, r=0),
            height=380,
            paper_bgcolor='rgba(15,20,40,0.95)',
        )
        return fig
    except Exception:
        return None
