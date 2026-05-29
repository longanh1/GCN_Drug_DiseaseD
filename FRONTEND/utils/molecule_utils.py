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


def smiles_to_3d_html_viewer(smiles: str, width: int = 620, height: int = 420, dark: bool = True) -> Optional[str]:
    """Generate an interactive 3D molecule viewer as a self-contained HTML page.

    Strategy (all browser-side, no RDKit Python needed):
      1. Embed the SMILES in the page.
      2. Load RDKit.js (WebAssembly) from CDN → generate a 3D molblock in the
         browser using ETKDG — works for *any* valid SMILES.
      3. Render the molblock with 3Dmol.js.
      4. If RDKit.js WASM fails, fall back to PubChem SDF fetched client-side.

    Returns an HTML string for ``st.components.v1.html()``, or None if smiles
    is empty.
    """
    if not smiles:
        return None

    # Escape the SMILES for safe embedding in a JS string literal
    smiles_js = (
        smiles
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("'", "\\'")
        .replace("\n", "")
        .replace("\r", "")
    )

    bg_css  = "#0a0e24" if dark else "#f8fafc"
    txt_css = "#e2e8f0" if dark else "#1e293b"
    btn_bg  = "rgba(99,102,241,0.15)" if dark else "rgba(99,102,241,0.08)"
    err_bg  = "rgba(244,63,94,0.12)"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<!-- 3Dmol.js: WebGL molecule renderer -->
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<!-- RDKit.js: WebAssembly port of RDKit — 3D coordinate generation in-browser -->
<script src="https://unpkg.com/@rdkit/rdkit/dist/RDKit_minimal.js"></script>
<style>
  *   {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:{bg_css}; font-family:system-ui,sans-serif; color:{txt_css}; }}
  #wrap {{ display:flex; flex-direction:column; width:{width}px; }}
  #viewer {{ width:{width}px; height:{height}px; position:relative;
             border-radius:10px; overflow:hidden; }}
  #status {{ font-size:11px; color:#94a3b8; text-align:center; padding:3px 0; min-height:18px; }}
  .btn-row {{ display:flex; flex-wrap:wrap; justify-content:center;
              gap:4px; padding:5px 4px; background:{bg_css}; }}
  button {{
    padding:3px 10px; font-size:11px; cursor:pointer;
    border:1px solid rgba(99,102,241,0.45); border-radius:6px;
    background:{btn_bg}; color:#818cf8; transition:background .15s;
  }}
  button:hover {{ background:rgba(99,102,241,0.32); }}
  #errbox {{ display:none; padding:8px 12px; margin:6px 4px;
             background:{err_bg}; border:1px solid rgba(244,63,94,0.3);
             border-radius:8px; font-size:12px; color:#fda4af; }}
</style>
</head>
<body>
<div id="wrap">
  <div id="viewer"></div>
  <div id="status">⏳ Đang khởi tạo RDKit.js...</div>
  <div class="btn-row">
    <button onclick="setStyle('ballstick')">Ball+Stick</button>
    <button onclick="setStyle('stick')">Stick</button>
    <button onclick="setStyle('sphere')">Sphere</button>
    <button onclick="setStyle('line')">Wire</button>
    <button onclick="toggleSpin()">⟳ Spin</button>
    <button onclick="viewer.zoomTo(); viewer.render();">Reset</button>
  </div>
  <div id="errbox"></div>
</div>
<script>
(function() {{
  var SMILES  = "{smiles_js}";
  var BG      = "{bg_css}";
  var viewer  = null;
  var spinning = false;
  var statusEl = document.getElementById('status');
  var errEl    = document.getElementById('errbox');

  function showErr(msg) {{
    errEl.textContent = msg;
    errEl.style.display = 'block';
    statusEl.textContent = '';
  }}

  function renderMolblock(molblock, source) {{
    try {{
      viewer = $3Dmol.createViewer(document.getElementById('viewer'), {{
        backgroundColor: BG
      }});
      viewer.addModel(molblock, 'sdf');
      applyStyle('ballstick');
      viewer.zoomTo();
      viewer.render();
      statusEl.textContent = '✅ 3D ' + source;
    }} catch(e) {{
      showErr('3Dmol render error: ' + e);
    }}
  }}

  function applyStyle(s) {{
    if (!viewer) return;
    if (s === 'ballstick') {{
      viewer.setStyle({{}}, {{stick:{{colorscheme:'Jmol',radius:0.12}},
                              sphere:{{colorscheme:'Jmol',scale:0.22}}}});
    }} else if (s === 'stick') {{
      viewer.setStyle({{}}, {{stick:{{colorscheme:'Jmol',radius:0.18}}}});
    }} else if (s === 'sphere') {{
      viewer.setStyle({{}}, {{sphere:{{colorscheme:'Jmol'}}}});
    }} else {{
      viewer.setStyle({{}}, {{line:{{colorscheme:'Jmol'}}}});
    }}
    viewer.render();
  }}

  window.setStyle = function(s) {{ applyStyle(s); }};
  window.toggleSpin = function() {{
    if (!viewer) return;
    spinning = !spinning;
    viewer.spin(spinning);
  }};

  // ── Strategy 1: RDKit.js WASM ────────────────────────────────────
  function tryRDKitWasm() {{
    statusEl.textContent = '⏳ RDKit.js: đang tạo tọa độ 3D...';
    initRDKitModule().then(function(RDKit) {{
      try {{
        var mol = RDKit.get_mol(SMILES);
        if (!mol || !mol.is_valid()) {{
          mol && mol.delete();
          throw new Error('SMILES không hợp lệ');
        }}
        mol.add_hs_in_place();
        var ok = mol.set_3d_coords();
        if (!ok) {{
          mol.delete();
          throw new Error('Không thể tạo tọa độ 3D');
        }}
        var molblock = mol.get_molblock();
        mol.delete();
        renderMolblock(molblock, '(RDKit.js WASM)');
      }} catch(e) {{
        statusEl.textContent = '⚠️ RDKit.js thất bại, thử PubChem...';
        tryPubChem();
      }}
    }}).catch(function() {{
      statusEl.textContent = '⚠️ RDKit.js không tải được, thử PubChem...';
      tryPubChem();
    }});
  }}

  // ── Strategy 2: PubChem API ───────────────────────────────────────
  function tryPubChem() {{
    statusEl.textContent = '⏳ Đang tải từ PubChem...';
    var encoded = encodeURIComponent(SMILES);
    fetch('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/' +
          encoded + '/cids/JSON')
      .then(function(r) {{ return r.json(); }})
      .then(function(data) {{
        var cids = (data.IdentifierList || {{}}).CID || [];
        if (!cids.length) throw new Error('Không tìm thấy trên PubChem');
        return fetch('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/' +
                     cids[0] + '/SDF?record_type=3d');
      }})
      .then(function(r) {{
        if (!r.ok) throw new Error('PubChem SDF 3D không có sẵn');
        return r.text();
      }})
      .then(function(sdf) {{
        renderMolblock(sdf, '(PubChem 3D)');
      }})
      .catch(function(e) {{
        showErr('⚠️ Không thể tạo cấu trúc 3D: ' + e.message +
                '. Vui lòng dùng chế độ 2D.');
      }});
  }}

  // Start with RDKit.js WASM
  if (typeof initRDKitModule === 'function') {{
    tryRDKitWasm();
  }} else {{
    // RDKit.js not yet loaded — wait a tick then try
    setTimeout(function() {{
      if (typeof initRDKitModule === 'function') {{
        tryRDKitWasm();
      }} else {{
        statusEl.textContent = '⚠️ RDKit.js chưa tải, thử PubChem...';
        tryPubChem();
      }}
    }}, 2000);
  }}
}})();
</script>
</body>
</html>"""
    return html


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

