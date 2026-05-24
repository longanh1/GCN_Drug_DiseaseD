"""Molecule visualization utilities using RDKit."""
import io
import base64
from typing import Optional

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# â”€â”€ CPK color palette â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ELEMENT_COLORS = {
    'C':  '#404040', 'H':  '#e8e8e8', 'O':  '#ff4136',
    'N':  '#0074d9', 'S':  '#ffdc00', 'F':  '#01ff70',
    'Cl': '#2ecc40', 'Br': '#85144b', 'P':  '#ff851b',
    'I':  '#7b0000', 'Si': '#b5b5b5', 'B':  '#ff69b4',
    'Fe': '#d44000', 'Cu': '#c87533', 'Zn': '#7d7d7d',
    'Ca': '#3ddc97', 'Mg': '#228b22', 'Na': '#ab00ab',
    'K':  '#8b00ff', 'Li': '#cc80ff', 'Se': '#ffa500',
}
ELEMENT_RADII = {
    'C': 14, 'H': 6,  'O': 16, 'N': 15, 'S': 18,
    'F': 12, 'Cl': 17,'Br': 18,'P':  17,'I':  20,
    'Si': 16,'B':  12, 'Fe': 18,'Cu': 17,'Zn': 16,
}
BOND_COLORS = {
    'SINGLE':    'rgba(120,120,140,0.7)',
    'DOUBLE':    'rgba(99,102,241,0.85)',
    'TRIPLE':    'rgba(16,185,129,0.9)',
    'AROMATIC':  'rgba(251,146,60,0.8)',
}


def smiles_to_image_b64(smiles: str, size: tuple = (400, 280)) -> Optional[str]:
    """Convert SMILES to high-quality base64 PNG with atom mapping colors."""
    if not RDKIT_AVAILABLE or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        opts = drawer.drawOptions()
        opts.addStereoAnnotation   = True
        opts.addAtomIndices        = False
        opts.addBondIndices        = False
        opts.atomLabelFontSize     = 0.45
        opts.bondLineWidth         = 2.0
        opts.multipleBondOffset    = 0.18
        opts.highlightRadius       = 0.3
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        data = drawer.GetDrawingText()
        return base64.b64encode(data).decode("utf-8")
    except Exception:
        pass
    # fallback: PIL draw
    try:
        mol = Chem.MolFromSmiles(smiles)
        img = Draw.MolToImage(mol, size=size, kekulize=True)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


def smiles_to_svg(smiles: str, size: tuple = (420, 300), dark_bg: bool = False) -> Optional[str]:
    """Convert SMILES to rich SVG with proper bond types and dark-mode support."""
    if not RDKIT_AVAILABLE or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        AllChem.Compute2DCoords(mol)

        drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        opts = drawer.drawOptions()
        opts.addStereoAnnotation = True
        opts.bondLineWidth        = 2.2
        opts.multipleBondOffset   = 0.18
        opts.atomLabelFontSize    = 0.48
        if dark_bg:
            opts.backgroundColour = (0.06, 0.08, 0.15, 1.0)
            opts.atomLabelFontSize = 0.5
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        if dark_bg:
            # lighten atom labels on dark background
            svg = svg.replace("fill:#000000", "fill:#e2e8f0")
            svg = svg.replace("stroke:#000000", "stroke:#cbd5e1")
        return svg
    except Exception:
        return None


def get_mol_properties(smiles: str) -> dict:
    """Extract comprehensive molecular properties."""
    if not RDKIT_AVAILABLE or not smiles:
        return {}
    try:
        from rdkit.Chem import Descriptors, rdMolDescriptors, QED
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        mw = rdMolDescriptors.CalcExactMolWt(mol)
        return {
            "num_atoms":    mol.GetNumAtoms(),
            "num_bonds":    mol.GetNumBonds(),
            "mol_weight":   round(mw, 3),
            "num_rings":    rdMolDescriptors.CalcNumRings(mol),
            "num_aromatic": rdMolDescriptors.CalcNumAromaticRings(mol),
            "num_hba":      rdMolDescriptors.CalcNumHBA(mol),
            "num_hbd":      rdMolDescriptors.CalcNumHBD(mol),
            "num_rotatable": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "tpsa":         round(rdMolDescriptors.CalcTPSA(mol), 2),
            "logp":         round(Descriptors.MolLogP(mol), 3),
            "qed":          round(QED.qed(mol), 4),
            "formula":      rdMolDescriptors.CalcMolFormula(mol),
            "heavy_atoms":  mol.GetNumHeavyAtoms(),
        }
    except Exception:
        return {}


def get_lipinski_analysis(smiles: str) -> dict:
    """Lipinski Rule-of-Five analysis for drug-likeness."""
    props = get_mol_properties(smiles)
    if not props:
        return {}
    mw     = props.get("mol_weight", 0)
    logp   = props.get("logp", 0)
    hbd    = props.get("num_hbd", 0)
    hba    = props.get("num_hba", 0)
    tpsa   = props.get("tpsa", 0)
    rot    = props.get("num_rotatable", 0)
    rules = {
        "MW â‰¤ 500":    mw   <= 500,
        "LogP â‰¤ 5":    logp <= 5,
        "HBD â‰¤ 5":     hbd  <= 5,
        "HBA â‰¤ 10":    hba  <= 10,
        "TPSA â‰¤ 140":  tpsa <= 140,
        "RotBonds â‰¤ 10": rot <= 10,
    }
    return {
        "rules":       rules,
        "passed":      sum(rules.values()),
        "total":       len(rules),
        "drug_like":   sum(rules.values()) >= 4,
    }


def smiles_to_3d_plotly(smiles: str, title: str = "", dark: bool = True):
    """Generate rich interactive 3D molecule figure with bond types and CPK colors."""
    if not RDKIT_AVAILABLE or not smiles:
        return None
    try:
        import plotly.graph_objects as go
        from rdkit.Chem import rdchem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mol_h = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
        if result != 0:
            result = AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())
        if result != 0:
            result = AllChem.EmbedMolecule(mol_h, randomSeed=42)
        if result != 0:
            return None

        try:
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
        except Exception:
            pass

        conf = mol_h.GetConformer()
        bg = 'rgba(10,14,36,0.97)' if dark else 'rgba(248,250,252,0.97)'
        txt_color = '#e2e8f0' if dark else '#1e293b'

        # â”€â”€ Group bonds by type â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        bond_traces = {}
        bond_type_map = {
            rdchem.BondType.SINGLE:   ('SINGLE',   3.5),
            rdchem.BondType.DOUBLE:   ('DOUBLE',   5.0),
            rdchem.BondType.TRIPLE:   ('TRIPLE',   6.5),
            rdchem.BondType.AROMATIC: ('AROMATIC', 4.0),
        }
        for bond in mol_h.GetBonds():
            btype, bwidth = bond_type_map.get(bond.GetBondType(), ('SINGLE', 3))
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            pi = conf.GetAtomPosition(i)
            pj = conf.GetAtomPosition(j)
            if btype not in bond_traces:
                bond_traces[btype] = {'x': [], 'y': [], 'z': [], 'w': bwidth}
            bond_traces[btype]['x'] += [pi.x, pj.x, None]
            bond_traces[btype]['y'] += [pi.y, pj.y, None]
            bond_traces[btype]['z'] += [pi.z, pj.z, None]

        # â”€â”€ Atom positions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        heavy_x, heavy_y, heavy_z = [], [], []
        heavy_colors, heavy_sizes, heavy_labels = [], [], []
        h_x, h_y, h_z = [], [], []

        for atom in mol_h.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            sym = atom.GetSymbol()
            if sym == 'H':
                h_x.append(pos.x); h_y.append(pos.y); h_z.append(pos.z)
            else:
                heavy_x.append(pos.x); heavy_y.append(pos.y); heavy_z.append(pos.z)
                heavy_colors.append(ELEMENT_COLORS.get(sym, '#888888'))
                heavy_sizes.append(ELEMENT_RADII.get(sym, 12))
                heavy_labels.append(sym)

        fig = go.Figure()

        # Bond traces
        bond_labels_map = {
            'SINGLE': 'Single bond', 'DOUBLE': 'Double bond',
            'TRIPLE': 'Triple bond', 'AROMATIC': 'Aromatic bond',
        }
        for btype, bd in bond_traces.items():
            fig.add_trace(go.Scatter3d(
                x=bd['x'], y=bd['y'], z=bd['z'],
                mode='lines',
                line=dict(color=BOND_COLORS.get(btype, 'rgba(160,160,160,0.6)'),
                          width=bd['w']),
                hoverinfo='none',
                showlegend=True,
                name=bond_labels_map.get(btype, btype),
            ))

        # Hydrogen atoms (small, semi-transparent)
        if h_x:
            fig.add_trace(go.Scatter3d(
                x=h_x, y=h_y, z=h_z,
                mode='markers',
                marker=dict(color='rgba(220,220,230,0.4)', size=4,
                            line=dict(width=0)),
                name='H atoms', hoverinfo='skip', showlegend=True,
            ))

        # Heavy atoms
        fig.add_trace(go.Scatter3d(
            x=heavy_x, y=heavy_y, z=heavy_z,
            mode='markers+text',
            marker=dict(
                color=heavy_colors,
                size=heavy_sizes,
                opacity=0.95,
                line=dict(width=1.2, color='rgba(255,255,255,0.3)'),
                symbol='circle',
            ),
            text=heavy_labels,
            textfont=dict(size=10, color=txt_color, family='Arial Black'),
            textposition='middle center',
            hovertemplate=(
                '<b style="color:#818cf8">%{text}</b><br>'
                'x = %{x:.3f} Ã…<br>'
                'y = %{y:.3f} Ã…<br>'
                'z = %{z:.3f} Ã…<extra></extra>'
            ),
            name='Heavy atoms',
            showlegend=True,
        ))

        axis_style = dict(
            showgrid=False, zeroline=False, showticklabels=False,
            showbackground=True,
            backgroundcolor='rgba(255,255,255,0.03)' if dark else 'rgba(0,0,0,0.02)',
            showline=False, title='',
        )
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>" if title else "",
                font=dict(size=13, color='#818cf8' if dark else '#4338ca'),
                x=0.5,
            ),
            scene=dict(
                xaxis=axis_style, yaxis=axis_style, zaxis=axis_style,
                bgcolor=bg,
                camera=dict(eye=dict(x=1.4, y=1.4, z=0.8)),
                aspectmode='data',
            ),
            legend=dict(
                font=dict(size=10, color=txt_color),
                bgcolor='rgba(0,0,0,0.3)' if dark else 'rgba(255,255,255,0.8)',
                bordercolor='rgba(99,102,241,0.3)',
                borderwidth=1,
                x=0.01, y=0.99,
            ),
            margin=dict(t=40, b=10, l=10, r=10),
            height=420,
            paper_bgcolor=bg,
            plot_bgcolor=bg,
        )
        return fig
    except Exception:
        return None


def smiles_to_2d_plotly(smiles: str, title: str = "", dark: bool = True):
    """Generate rich 2D molecule plot using RDKit 2D coordinates via Plotly."""
    if not RDKIT_AVAILABLE or not smiles:
        return None
    try:
        import plotly.graph_objects as go
        from rdkit.Chem import rdchem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()

        bg = 'rgba(10,14,36,0.97)' if dark else 'rgba(248,250,252,0.97)'
        txt_color = '#e2e8f0' if dark else '#1e293b'

        bond_type_map = {
            rdchem.BondType.SINGLE:   ('SINGLE',   2.0),
            rdchem.BondType.DOUBLE:   ('DOUBLE',   3.5),
            rdchem.BondType.TRIPLE:   ('TRIPLE',   4.5),
            rdchem.BondType.AROMATIC: ('AROMATIC', 2.8),
        }

        bond_segs: dict = {}
        for bond in mol.GetBonds():
            btype, bwidth = bond_type_map.get(bond.GetBondType(), ('SINGLE', 2))
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            pi = conf.GetAtomPosition(i)
            pj = conf.GetAtomPosition(j)
            if btype not in bond_segs:
                bond_segs[btype] = {'x': [], 'y': [], 'w': bwidth}
            bond_segs[btype]['x'] += [pi.x, pj.x, None]
            bond_segs[btype]['y'] += [pi.y, pj.y, None]

        xs, ys, colors, sizes, labels = [], [], [], [], []
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            sym = atom.GetSymbol()
            xs.append(pos.x); ys.append(pos.y)
            colors.append(ELEMENT_COLORS.get(sym, '#888888'))
            sizes.append(max(18, ELEMENT_RADII.get(sym, 14) * 1.4))
            labels.append(sym if sym != 'C' else '')

        fig = go.Figure()
        for btype, bd in bond_segs.items():
            fig.add_trace(go.Scatter(
                x=bd['x'], y=bd['y'], mode='lines',
                line=dict(color=BOND_COLORS.get(btype, 'rgba(140,140,160,0.7)'),
                          width=bd['w']),
                hoverinfo='none', showlegend=True, name=btype.capitalize(),
            ))

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode='markers+text',
            marker=dict(
                color=colors, size=sizes, opacity=0.95,
                line=dict(width=1.5, color='rgba(255,255,255,0.4)'),
            ),
            text=labels,
            textfont=dict(size=11, color=txt_color, family='Arial Black'),
            textposition='middle center',
            hovertemplate='<b>%{text}</b><extra></extra>',
            name='Atoms', showlegend=True,
        ))

        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>" if title else "",
                font=dict(size=13, color='#818cf8' if dark else '#4338ca'),
                x=0.5,
            ),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False,
                       scaleanchor='x'),
            legend=dict(
                font=dict(size=10, color=txt_color),
                bgcolor='rgba(0,0,0,0.35)' if dark else 'rgba(255,255,255,0.85)',
                bordercolor='rgba(99,102,241,0.3)',
                borderwidth=1,
            ),
            margin=dict(t=40, b=10, l=10, r=10),
            height=360,
            paper_bgcolor=bg,
            plot_bgcolor=bg,
        )
        return fig
    except Exception:
        return None

