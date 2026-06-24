"""
prompt_browser -- Streamlit front end.

Browse/filter generations from PostgreSQL, pick one, tweak seed/prompt, and
re-submit the stored ComfyUI workflow to generate.

Run:  streamlit run app.py
"""
import os

import streamlit as st
from dotenv import load_dotenv

import db
from comfy_client import ComfyClient
import graph_patch

load_dotenv()
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")

st.set_page_config(page_title="Prompt Browser", layout="wide")


@st.cache_resource
def get_conn():
    return db.connect()


def q(sql, params=None, one=False):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(sql, params or ())
        cols = [c[0] for c in cur.description] if cur.description else []
        rows = cur.fetchall()
    conn.commit()
    dicts = [dict(zip(cols, r)) for r in rows]
    return (dicts[0] if dicts else None) if one else dicts


@st.cache_data(ttl=60)
def distinct(col):
    return [r[col] for r in q(
        "SELECT DISTINCT %s AS %s FROM generations WHERE %s IS NOT NULL ORDER BY 1"
        % (col, col, col))]


@st.cache_data(ttl=60)
def lora_names():
    return [r["name"] for r in q("SELECT name FROM loras ORDER BY name")]


# --------------------------------------------------------------------------- #
# Sidebar filters
# --------------------------------------------------------------------------- #
st.sidebar.header("Filters")
f_family = st.sidebar.multiselect("Model family", distinct("model_family"))
f_type = st.sidebar.multiselect("Gen type", distinct("gen_type"))
f_image = st.sidebar.multiselect("Image type", distinct("image_type"))
f_lora = st.sidebar.selectbox("Uses lora", ["(any)"] + lora_names())
f_text = st.sidebar.text_input("Prompt contains")
limit = st.sidebar.slider("Max results", 10, 500, 100, step=10)

where, params = [], []
if f_family:
    where.append("model_family = ANY(%s)"); params.append(f_family)
if f_type:
    where.append("gen_type = ANY(%s)"); params.append(f_type)
if f_image:
    where.append("image_type = ANY(%s)"); params.append(f_image)
if f_text:
    where.append("positive_prompt ILIKE %s"); params.append("%" + f_text + "%")
if f_lora != "(any)":
    where.append("g.id IN (SELECT gl.generation_id FROM generation_loras gl "
                 "JOIN loras l ON l.id = gl.lora_id WHERE l.name = %s)")
    params.append(f_lora)

clause = ("WHERE " + " AND ".join(where)) if where else ""
rows = q(
    "SELECT g.id, model_family, gen_type, image_type, seed, width, height, "
    "lora_count, positive_prompt FROM generations g %s ORDER BY g.id DESC LIMIT %s"
    % (clause, "%s"), params + [limit])

st.title("🎨 Prompt Browser")
st.caption("%d result(s) — ComfyUI: %s" % (len(rows), COMFYUI_URL))

if not rows:
    st.info("No matches. Loosen the filters, or run build_db.py to import data.")
    st.stop()

# Compact table
st.dataframe(
    [{"id": r["id"], "family": r["model_family"], "type": r["gen_type"],
      "rating": r["image_type"], "loras": r["lora_count"],
      "prompt": (r["positive_prompt"] or "")[:120]} for r in rows],
    use_container_width=True, hide_index=True)

ids = [r["id"] for r in rows]
sel_id = st.selectbox("Select a generation id", ids)
gen = q("SELECT * FROM generations WHERE id=%s", (sel_id,), one=True)
gen_loras = q(
    "SELECT l.name, gl.strength FROM generation_loras gl "
    "JOIN loras l ON l.id=gl.lora_id WHERE gl.generation_id=%s ORDER BY l.name",
    (sel_id,))

# --------------------------------------------------------------------------- #
# Detail + generate
# --------------------------------------------------------------------------- #
left, right = st.columns([3, 2])
with left:
    st.subheader("Prompt")
    new_prompt = st.text_area("Positive prompt", gen["positive_prompt"] or "", height=160)
    if gen["negative_prompt"]:
        st.text_area("Negative prompt (read-only)", gen["negative_prompt"], height=80, disabled=True)
    st.write("**Loras:** " + (", ".join("%s@%s" % (l["name"], l["strength"]) for l in gen_loras) or "none"))

with right:
    st.subheader("Settings")
    st.write({k: gen[k] for k in ("model_family", "model_file", "gen_type",
                                  "image_type", "width", "height", "steps",
                                  "sampler", "scheduler", "guidance", "denoise")})
    randomize = st.checkbox("Randomize seed", value=True)
    seed_in = st.text_input("Seed (used if not randomizing)", str(gen["seed"] or 0))
    override_prompt = st.checkbox("Use my edited prompt", value=False,
                                  help="Off = replay the original prompt exactly.")

if st.button("🚀 Generate", type="primary", use_container_width=True):
    if not gen.get("workflow"):
        st.error("This generation has no stored workflow graph; cannot replay.")
        st.stop()
    graph, info = graph_patch.prepare(
        gen["workflow"],
        new_seed=None if randomize else seed_in,
        randomize=randomize,
        positive_prompt=new_prompt if override_prompt else None,
    )
    if override_prompt and info["prompt_set"] is False:
        st.warning("Couldn't locate an editable prompt node (e.g. a composed "
                   "multi-prompt workflow); generating with the original prompt.")
    st.caption("Seed: %s" % info["seed"])
    client = ComfyClient(COMFYUI_URL)
    bar = st.progress(0.0, text="Queued…")
    try:
        def on_progress(v, m):
            bar.progress(min(v / m, 1.0) if m else 0.0, text="Sampling %d/%d" % (v, m))
        pid = client.queue(graph)
        client.wait(pid, on_progress)
        imgs = client.images(pid)
        bar.progress(1.0, text="Done")
        if not imgs:
            st.warning("Finished but no images returned.")
        for name, data in imgs:
            st.image(data, caption=name, use_container_width=True)
    except Exception as e:
        st.error("Generation failed: %s" % e)
