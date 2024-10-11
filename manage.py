import os, json
from flask import Flask, request, render_template, jsonify
from lib.GraphManageEngine import GraphManage

GM_DIR = r"./static/database/graphDB"
if (not GM_DIR is None) and not os.path.exists(GM_DIR):
    os.mkdir(GM_DIR)
GM = GraphManage(dir_path=GM_DIR)

app = Flask(__name__, template_folder="./templates", static_folder="./static", static_url_path='/static')

app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.variable_start_string = '||<-EL PSY CONGROO->||'
app.jinja_env.variable_end_string = '||<-EL PSY CONGROO->||'

@ app.route('/graph-interface/', methods=['GET','POST'])
def graph_interface_init():
    return render_template("graph-interface.html")

@ app.route('/graph-interface/save-graph', methods=['GET','POST'])
def graph_interface_graph_save():
    GM.save()
    return ""

@ app.route('/graph-interface/flush-graph', methods=['GET','POST'])
def graph_interface_graph_flush():
    GM.flushdb()
    return jsonify({"nodes": [], "edges": []})
    
@ app.route('/graph-interface/load-graph', methods=['GET','POST'])
def graph_interface_graph_init():
    ## TODO - reload step
    # struct = {}
    # GraphManage.restart(nodes=struct.get("nodes", []), edges=struct.get("edges", []))
    GM.load()
    
    names = json.loads(request.form.get("names") or "[]")
    node_types = json.loads(request.form.get("nodeTypes") or "[]")
    conn_types = json.loads(request.form.get("connTypes") or "[]")
    mode = str(request.form.get("mode") or "bi")
    n = int(request.form.get("n") or 5)
    nodes, edges = GM.get_n_connect_map(names=names,
                                        conn_types=conn_types,
                                        node_types=node_types,
                                        name_typ="uuid",
                                        mode=mode, n=n)
    return jsonify({"nodes": nodes, "edges": edges})

@ app.route('/graph-interface/change-query', methods=['GET','POST'])
def graph_interface_change_query():
    names = json.loads(request.form.get("names") or "[]")
    conn_types = json.loads(request.form.get("connTypes") or "[]")
    node_types = json.loads(request.form.get("nodeTypes") or "[]")
    mode = str(request.form.get("mode") or "bi")
    n = int(request.form.get("n") or 5)
    nodes, edges = GM.get_n_connect_map(names=names,
                                        conn_types=conn_types,
                                        node_types=node_types,
                                        name_typ="uuid",
                                        mode=mode, n=n)
    return jsonify({"nodes": nodes, "edges": edges})

@ app.route('/graph-interface/add-edge', methods=['GET','POST'])
def graph_interface_add_edge():
    edge = json.loads(request.form.get("edge") or "{}")
    old_type = request.form.get("oldType")
    if old_type:
        GM.edit_edge(edge, old_type=old_type)
    else:
        GM.add_edge(edge)
    
    names = json.loads(request.form.get("names") or "[]")
    conn_types = json.loads(request.form.get("connTypes") or "[]")
    node_types = json.loads(request.form.get("nodeTypes") or "[]")
    mode = str(request.form.get("mode") or "bi")
    n = int(request.form.get("n") or 5)
    nodes, edges = GM.get_n_connect_map(names=names,
                                                 conn_types=conn_types,
                                                 node_types=node_types,
                                                 name_typ="uuid",
                                                 mode=mode, n=n)
    return jsonify({"nodes": nodes, "edges": edges})

@ app.route('/graph-interface/del-edge', methods=['GET','POST'])
def graph_interface_del_edge():
    edge = json.loads(request.form.get("edge") or "{}")
    GM.del_edge(edge)
    
    names = json.loads(request.form.get("names") or "[]")
    conn_types = json.loads(request.form.get("connTypes") or "[]")
    node_types = json.loads(request.form.get("nodeTypes") or "[]")
    mode = str(request.form.get("mode") or "bi")
    n = int(request.form.get("n") or 5)
    nodes, edges = GM.get_n_connect_map(names=names,
                                                 conn_types=conn_types,
                                                 node_types=node_types,
                                                 name_typ="uuid",
                                                 mode=mode, n=n)
    return jsonify({"nodes": nodes, "edges": edges})

@ app.route('/graph-interface/edit-edge', methods=['GET','POST'])
def graph_interface_edit_edge():
    edge = json.loads(request.form.get("edge") or "{}")
    old_type = request.form.get("oldType")
    GM.edit_edge(edge, old_type)

    old_type = json.loads(request.form.get("oldType") or "{}")
    names = json.loads(request.form.get("names") or "[]")
    conn_types = json.loads(request.form.get("connTypes") or "[]")
    node_types = json.loads(request.form.get("nodeTypes") or "[]")
    mode = str(request.form.get("mode") or "bi")
    n = int(request.form.get("n") or 5)
    nodes, edges = GM.get_n_connect_map(names=names,
                                                 conn_types=conn_types,
                                                 node_types=node_types,
                                                 name_typ="uuid",
                                                 mode=mode, n=n)
    return jsonify({"nodes": nodes, "edges": edges})

@ app.route('/graph-interface/add-node', methods=['GET','POST'])
def graph_interface_add_node():
    node = json.loads(request.form.get("node") or "{}")
    GM.add_node(node)

    names = json.loads(request.form.get("names") or "[]")
    conn_types = json.loads(request.form.get("connTypes") or "[]")
    node_types = json.loads(request.form.get("nodeTypes") or "[]")
    mode = str(request.form.get("mode") or "bi")
    n = int(request.form.get("n") or 5)
    nodes, edges = GM.get_n_connect_map(names=names,
                                        conn_types=conn_types,
                                        node_types=node_types,
                                        name_typ="uuid",
                                        mode=mode, n=n)
    return jsonify({"nodes": nodes, "edges": edges})

@ app.route('/graph-interface/fuse-node', methods=['GET','POST'])
def graph_interface_fuse_node():
    node = json.loads(request.form.get("node") or "{}")
    targ_uid = request.form.get("targ_uid") or ""
    GM.fuse_node(node, targ_uid)

    names = json.loads(request.form.get("names") or "[]")
    conn_types = json.loads(request.form.get("connTypes") or "[]")
    node_types = json.loads(request.form.get("nodeTypes") or "[]")
    mode = str(request.form.get("mode") or "bi")
    n = int(request.form.get("n") or 5)
    nodes, edges = GM.get_n_connect_map(names=names,
                                        conn_types=conn_types,
                                        node_types=node_types,
                                        name_typ="uuid",
                                        mode=mode, n=n)
    return jsonify({"nodes": nodes, "edges": edges})

@ app.route('/graph-interface/del-node', methods=['GET','POST'])
def graph_interface_del_node():
    node = json.loads(request.form.get("node") or "{}")
    GM.del_node(node)

    names = json.loads(request.form.get("names") or "[]")
    conn_types = json.loads(request.form.get("connTypes") or "[]")
    node_types = json.loads(request.form.get("nodeTypes") or "[]")
    mode = str(request.form.get("mode") or "bi")
    n = int(request.form.get("n") or 5)
    nodes, edges = GM.get_n_connect_map(names=names,
                                                 conn_types=conn_types,
                                                 node_types=node_types,
                                                 name_typ="uuid",
                                                 mode=mode, n=n)
    return jsonify({"nodes": nodes, "edges": edges})

@ app.route('/graph-interface/edit-node', methods=['GET','POST'])
def graph_interface_edit_node():
    node = json.loads(request.form.get("node") or "{}")
    GM.edit_node(node)

    old_type = json.loads(request.form.get("oldType") or "{}")
    names = json.loads(request.form.get("names") or "[]")
    conn_types = json.loads(request.form.get("connTypes") or "[]")
    node_types = json.loads(request.form.get("nodeTypes") or "[]")
    mode = str(request.form.get("mode") or "bi")
    n = int(request.form.get("n") or 5)
    nodes, edges = GM.get_n_connect_map(names=names,
                                                 conn_types=conn_types,
                                                 node_types=node_types,
                                                 name_typ="uuid",
                                                 mode=mode, n=n)
    return jsonify({"nodes": nodes, "edges": edges})

@ app.route('/graph-interface/get-node-info', methods=['GET','POST'])
def graph_interface_get_node_info():
    node_uid = request.form.get("nodeUid") or ""
    res = GM.get_node_info(node_uid)
    return jsonify(res)

@ app.route('/graph-interface/get-edge-info', methods=['GET','POST'])
def graph_interface_get_edge_info():
    srcUid = request.form.get("srcUid") or ""
    trgUid = request.form.get("trgUid") or ""
    conn_type = request.form.get("connType") or ""
    res = GM.get_edge_info(srcUid, trgUid, conn_type)
    return jsonify(res)

@ app.route("/graph-interface/re-match-name", methods=['GET','POST'])
def graph_interface_get_match_name():
    query = request.form.get("query") or ""
    nlimit = int(request.form.get("nlimit") or 20)
    res = GM.get_match_name(query, nlimit)
    return jsonify(res)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8090)
