import os, math, uuid, json, re, time, shutil
from typing import Dict
import numpy as np
from scipy.sparse import coo_matrix

UIDKEY = "id"
NAMEKEY = "name"
NODETYPKEY = "class"
CONNTYPKEY = "class"
SRCKEY = "source"
TARGKEY = "target"
NODEPROPKEY = "properties"
CONNPROPKEY = "properties"

## N-SKIP SEARCH ALGORITHM
def _get_log_map(num):
    if num == 0:
        return []
    else:
        n_log = int(math.log2(num))
        return _get_log_map(num - 2**n_log) + [n_log]

def _n_skip_forward_connect(matrix, id_matrix=None, n=5):
    """
    [Calcul n-skip Adjacency Matrix]
    Boolean Operation:
      a) True + True = True
      b) False + True = True + False = True
      c) False + False = False
    
    => Boolean Matrix for n-skip connection:
          A[n] = A[1]^n      with:
              a) A[1] = I + A (Just to guarantee IDentity connection)
              b) A -> Adjacency Matrix from EDGES-list
    """
    if id_matrix is None:
        id_matrix = coo_matrix(([], ([], [])),
                               shape=matrix.shape, dtype=bool).todok()
    result = id_matrix.copy()
    
    if n == 0:
        return result
    
    result_immediate = matrix + id_matrix
    n_log_s = _get_log_map(n)
    for i in range(n_log_s[-1]):
        if i in n_log_s:
            result = result.dot(result_immediate)
        result_immediate = result_immediate.dot(result_immediate)
    result = result.dot(result_immediate)
    
    # result.eliminate_zeros()
    return result

def _n_skip_backward_connect(matrix, id_matrix, n=5):
    return _n_skip_forward_connect(matrix.T, id_matrix, n=n)

def _n_skip_bilateral_connect(matrix, id_matrix, n=5):
    return _n_skip_forward_connect(matrix, id_matrix, n=n) + _n_skip_backward_connect(matrix.T, id_matrix, n=n)

def _get_nonzero_by_row(matrix, rows):
    _matrix = matrix.tocsr()[rows, :]
    _matrix.eliminate_zeros()
    return sorted(set(_matrix.indices))

## Uuid generation
def get_uuid(name=""):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{name}#%.7f"%time.time()))
    
class GraphManage:
    def __init__(self, nodes=[], edges=[], dir_path=None, demo_top_k=5):
        if not(dir_path is None) and not os.path.exists(dir_path):
            os.mkdir(dir_path)
        self._dir_path = dir_path
        self._demo_top_k = max(int(demo_top_k), 1)
        self.restart(nodes=nodes, edges=edges)
        
    def flushdb(self):
        if os.path.exists(self._dir_path):
            shutil.rmtree(self._dir_path)
        os.mkdir(self._dir_path)
        self.restart(nodes=[], edges=[])
        
    def get_match_name(self, query, nlimit=20):
        res = []
        for node in self._full_nodes:
            name = node[NAMEKEY]
            if not re.search(query, name) is None:
                res.append(node)
            if len(res) >= nlimit:
                break
        return res

    ## DEMO NODE OPERATION
    @ staticmethod
    def _log_node_properties(node, dirn):
        fname = os.path.join(dirn, f"node_property_{node[UIDKEY]}.gmdb4nd")
        if dirn is None or (not os.path.exists(dirn)):
            return node
        else:
            with open(fname, "w+", encoding="utf-8") as f:
                json.dump(node.get(NODEPROPKEY, {}), f, indent=4, ensure_ascii=False)
            return {**node, NODEPROPKEY: fname}

    @ staticmethod
    def _read_node_properties(node, dirn):
        if NODEPROPKEY in node.keys() and isinstance(node[NODEPROPKEY], Dict):
            return node[NODEPROPKEY]
        if (not UIDKEY not in node.keys()) or (not node[UIDKEY].replace(" ", "")):
            return {}
        if dirn is None or (not os.path.exists(dirn)):
            return {}
        fname = os.path.join(dirn, f"node_property_{node[UIDKEY]}.gmdb4nd")
        if not os.path.exists(fname):
            return {}
        with open(fname, "r+", encoding="utf-8") as f:
            res = json.load(f)
        return res

    def _fuse_node_properties(self, node_rem, node_del, dirn):
        _node_properties_del = self._read_node_properties(node_del, dirn)
        _node_properties_rem = self._read_node_properties(node_rem, dirn)
        properties = {**_node_properties_del, **_node_properties_rem}
        return self._log_node_properties({
            **node_rem,
            **node_del,
            UIDKEY: node_rem[UIDKEY],
            NODEPROPKEY: properties
            },dirn)

    @ staticmethod
    def _del_node_properties(node, dirn):
        fname = os.path.join(dirn, f"node_property_{node[UIDKEY]}.gmdb4nd")
        if dirn is None or (not os.path.exists(dirn)):
            return 
        if os.path.exists(fname):
            os.remove(fname)

    @ staticmethod
    ## DEMO EDGE OPERATION
    def _log_edge_properties(edge, dirn):
        srcUid = edge[SRCKEY]
        trgUid = edge[TARGKEY]
        conn_typ = edge.get(CONNTYPKEY, "unlabeled")
        edgeRef = f"SRC@{srcUid}-TARG@{trgUid}-TYPE@{conn_typ}"
        if (not dirn is None) and (os.path.exists(dirn)):
            fname = os.path.join(dirn, f"edge_property_{edgeRef}.gmdb4ed")
            with open(fname, "w+", encoding="utf-8") as f:
                json.dump(edge.get(CONNPROPKEY, {}), f, indent=4, ensure_ascii=False)
            return {**edge, CONNPROPKEY: f"edge_property_{edgeRef}.gmdb4ed"}
        else:
            return edge

    @ staticmethod
    def _read_edge_properties(edge, dirn):
        if CONNPROPKEY in edge.keys() and isinstance(edge[CONNPROPKEY], Dict):
            return edge[CONNPROPKEY]
        srcUid = edge[SRCKEY]
        trgUid = edge[TARGKEY]
        conn_typ = edge.get(CONNTYPKEY, "unlabeled")
        edgeRef = f"SRC@{srcUid}-TARG@{trgUid}-TYPE@{conn_typ}"
        if dirn is None or (not os.path.exists(dirn)):
            return {}
        fname = os.path.join(dirn, f"edge_property_{edgeRef}.gmdb4ed")
        if not os.path.exists(fname):
            return {}
        with open(fname, "r+", encoding="utf-8") as f:
            res = json.load(f)
        return res

    def _fuse_edge_properties(self, edge_rem, edge_del, dirn):
        _edge_properties_del = self._read_edge_properties(edge_del, dirn)
        _edge_properties_rem = self._read_edge_properties(edge_rem, dirn)
        properties = {**_edge_properties_del, **_edge_properties_rem}
        return self._log_edge_properties({
            **edge_rem,
            **edge_del,
            SRCKEY: edge_rem[SRCKEY],
            TARGKEY: edge_rem[TARGKEY],
            CONNPROPKEY: properties
            },dirn)

    @ staticmethod
    def _del_edge_properties(edge, dirn):
        srcUid = edge[SRCKEY]
        trgUid = edge[TARGKEY]
        conn_typ = edge.get(CONNTYPKEY, "unlabeled")
        edgeRef = f"SRC@{srcUid}-TARG@{trgUid}-TYPE@{conn_typ}"
        if dirn is None or (not os.path.exists(dirn)):
            return 
        fname = os.path.join(dirn, f"edge_property_{edgeRef}.gmdb4ed")
        if os.path.exists(fname):
            os.remove(fname)

    def _wash_node(self, node):
        if (not UIDKEY in node.keys()) or (not node[UIDKEY].replace(" ", "")):
            node[UIDKEY] = get_uuid(name=node[NAMEKEY])
        if (not self._dir_path is None) and isinstance(node.get(NODEPROPKEY, {}), Dict):
            node = self._log_node_properties(node, self._dir_path)
        return node

    def _wash_edge(self, edge):
        if (not self._dir_path is None) and isinstance(edge.get(CONNPROPKEY, {}), Dict):
            edge = self._log_edge_properties(edge, self._dir_path)
        return edge

    def restart(self, nodes=[], edges=[]):
        self._full_nodes = [self._wash_node(node) for node in nodes]
        self._full_edges = [self._wash_edge(edge) for edge in edges]
        
        self._remap_nodes()
        self._remap_edges()
        
        _full_edges_relations = {}
        for edge in edges:
            itm_typ = edge.get(CONNTYPKEY, "unlabeled")
            if not itm_typ in _full_edges_relations.keys():
                _full_edges_relations[itm_typ] = [[],[]]
            _full_edges_relations[itm_typ][0].append(self._full_nodeIds[edge[SRCKEY]])
            _full_edges_relations[itm_typ][1].append(self._full_nodeIds[edge[TARGKEY]])    
        self._full_edges_matrix = {k: coo_matrix(([True for _ in range(len(v[0]))],
                                                  (v[0],v[1])),
                                                 shape=(self._full_num_nodes,
                                                        self._full_num_nodes), dtype=bool).todok()
                                   for k, v in _full_edges_relations.items()}
        self._full_identity_matrix = self._gen_ind_matrix(self._full_num_nodes)

    @ property
    def struct(self):
        return {"nodes": self._full_nodes, "edges": self._full_edges}

    def _remap_nodes(self):
        self._full_nodeIds = {itm[UIDKEY]: itmId for itmId, itm in enumerate(self._full_nodes)}
        self._full_nodeNames = {itm[NAMEKEY]: itmId for itmId, itm in enumerate(self._full_nodes)}

    def _remap_edges(self):
        self._full_edgesIds = {
            f"SRC@{edge[SRCKEY]}-TARG@{edge[TARGKEY]}-TYPE@{edge.get(CONNTYPKEY, 'unlabeled')}": edgeId
            for edgeId, edge in enumerate(self._full_edges)}

    @ property
    def _full_num_nodes(self):
        return len(self._full_nodes)

    @ property
    def _full_num_edges(self):
        return len(self._full_edges)

    @ property
    def node_types(self):
        return sorted({node.get(NODETYPKEY, 'unlabeled') for node in self._full_nodes})

    @ property
    def conn_types(self):
        return sorted({edge.get(CONNTYPKEY, 'unlabeled') for edge in self._full_edges})

    @ staticmethod
    def _gen_ind_matrix(L):  
         return coo_matrix(([True for _ in range(L)], (range(L), range(L))),
                           shape=(L, L), dtype=bool).todok()

    def _get_top_k_nodeId(self, conn_types, node_types, k=5):
        if not node_types is None:
            if isinstance(node_types, str):
                node_types = [node_types]
            _node_inds = [ind for ind, node in enumerate(self._full_nodes) if node.get(NODETYPKEY, "unlabeled") in node_types]
        _node_inds = list(range(self._full_num_nodes))
        
        conn_matrix = self._full_identity_matrix[_node_inds, :][:, _node_inds].copy()
        for conn_typ in conn_types:
            if conn_typ in self._full_edges_matrix.keys():
                conn_matrix += self._full_edges_matrix[conn_typ][_node_inds, :][:, _node_inds]
        conn_matrix = conn_matrix.astype("int16")
        return (conn_matrix.sum(axis=0).A1*conn_matrix.sum(axis=1).A1).argsort()[::-1][:k]
        

    def get_n_connect_map(self, names=[], conn_types=None, node_types=None, name_typ="name", mode="bi", n=5):
        if conn_types is None:
            conn_types = list(self._full_edges_matrix.keys())
        elif isinstance(conn_types, str):
            conn_types = [conn_types]
            
        conn_matrix = self._full_identity_matrix.copy()
        for conn_typ in conn_types:
            if conn_typ in self._full_edges_matrix.keys():
                conn_matrix += self._full_edges_matrix[conn_typ]

        n = min(n, self._full_num_edges)
        if mode.lower().startswith("bi"):
            full_conn_matrix = _n_skip_bilateral_connect(conn_matrix, self._full_identity_matrix, n=n)
        elif mode.lower().startswith("b"):
            full_conn_matrix = _n_skip_backward_connect(conn_matrix, self._full_identity_matrix, n=n)
        else:
            full_conn_matrix = _n_skip_forward_connect(conn_matrix, self._full_identity_matrix, n=n)
            
        if isinstance(names, str) and names.replace(" ", ""):
            names = [names]
        else:
            names = self._get_top_k_nodeId(conn_types=conn_types, node_types=node_types, k=self._demo_top_k)
            name_typ = "nodeId"
        
        if name_typ == "name":
            names = [self._full_nodeNames[name] for name in names if name in self._full_nodeNames.keys()]
        elif name_typ == "uuid":
            names = [self._full_nodeIds[name] for name in names if name in self._full_nodeIds.keys()]
            
        node_inds = _get_nonzero_by_row(full_conn_matrix, names)
        if not node_types is None:
            if isinstance(node_types, str):
                node_types = [node_types]
            _nodesInds = [(self._full_nodes[ind], ind) for ind in node_inds if self._full_nodes[ind].get(NODETYPKEY, "unlabeled") in node_types]
            if len(_nodesInds) == 0:
                _nodes = []
                node_inds = []
            else:
                _nodes, node_inds = zip(*_nodesInds)
        else:
            _nodes = [self._full_nodes[ind] for ind in node_inds]
        _edges = [edge for edge in self._full_edges if edge.get(CONNTYPKEY, "unlabeled") in conn_types and \
                  self._full_nodeIds[edge[SRCKEY]] in node_inds and self._full_nodeIds[edge[TARGKEY]] in node_inds]
        return _nodes, _edges

    def add_edge(self, edge):
        srcUid = edge[SRCKEY]
        trgUid = edge[TARGKEY]
        srcId = self._full_nodeIds[srcUid]
        trgId = self._full_nodeIds[trgUid]
        conn_typ = edge.get(CONNTYPKEY, "unlabeled")
        edgeRef = f"SRC@{srcUid}-TARG@{trgUid}-TYPE@{conn_typ}"
        if edgeRef in self._full_edgesIds.keys():
            return self.edit_edge(edge)
        
        if not conn_typ in self._full_edges_matrix.keys():
            self._full_edges_matrix[conn_typ] = (self._full_identity_matrix * False).todok()
        self._full_edges_matrix[conn_typ][srcId, trgId] = True

        if (not self._dir_path is None):
            edge = self._log_edge_properties(edge, dirn=self._dir_path)
            
        self._full_edges.append(edge)
        self._full_edgesIds[edgeRef] = self._full_num_edges - 1
        
        return self._full_nodes, self._full_edges

    def del_edge(self, edge):
        srcUid = edge[SRCKEY]
        trgUid = edge[TARGKEY]
        srcId = self._full_nodeIds[srcUid]
        trgId = self._full_nodeIds[trgUid]
        conn_typ = edge.get(CONNTYPKEY, "unlabeled")
        
        self._full_edges_matrix[conn_typ][srcId, trgId] = False
        # self._full_edges_matrix[conn_typ].eliminate_zeros()
        
        edgeRef = f"SRC@{srcUid}-TARG@{trgUid}-TYPE@{conn_typ}"
        self._full_edges.pop(self._full_edgesIds[edgeRef])
        self._remap_edges()

        if (not self._dir_path is None):
            self._del_edge_properties(edge, dirn=self._dir_path)
                
        return self._full_nodes, self._full_edges
        
    def edit_edge(self, edge, old_type=None):
        srcUid = edge[SRCKEY]
        trgUid = edge[TARGKEY]
        srcId = self._full_nodeIds[srcUid]
        trgId = self._full_nodeIds[trgUid]

        new_type = edge.get(CONNTYPKEY, "unlabeled")
        old_type = old_type or new_type
        
        self._full_edges_matrix[old_type][srcId, trgId] = False
        # self._full_edges_matrix[old_type].eliminate_zeros()
        if not new_type in self._full_edges_matrix.keys():
            self._full_edges_matrix[new_type] = (self._full_identity_matrix * False).todok()
            # self._full_edges_matrix[conn_typ].eliminate_zeros()
        self._full_edges_matrix[new_type][srcId, trgId] = True
        
        old_edgeRef = f"SRC@{srcUid}-TARG@{trgUid}-TYPE@{old_type}"
        edgeId = self._full_edgesIds[old_edgeRef]

        if (not self._dir_path is None):
            self._full_edges[edgeId] = self._fuse_edge_properties(self._full_edges[edgeId], edge,
                                                                  dirn=self._dir_path)
        else:
            self._full_edges[edgeId] = edge

        if old_type != new_type:
            self._remap_edges()
            
        return self._full_nodes, self._full_edges

    @staticmethod
    def _add_null_line(m):
        if len(m.keys()) == 0:
            return coo_matrix(([], [[],[]]),
                          shape=[_dim+1 for _dim in m.shape], dtype=bool).todok()
        return coo_matrix((list(m.values()), [list(itm) for itm in zip(*m.keys())]),
                          shape=[_dim+1 for _dim in m.shape], dtype=bool).todok()

    def add_node(self, node):
        if (not UIDKEY in node.keys()) or (not node[UIDKEY].replace(" ", "")):
            node[UIDKEY] = get_uuid(name=node[NAMEKEY])
        elif node[UIDKEY] in self._full_nodeIds.keys():
            return self.edit_node(node)
        
        name_ref = node[NAMEKEY]; cons = 1
        while node[NAMEKEY] in self._full_nodeNames.keys():
            node[NAMEKEY] = f"{name_ref}_{cons}"
            cons += 1

        if (not self._dir_path is None):
            node = self._log_node_properties(node, dirn=self._dir_path)
        self._full_nodes.append(node)
        
        self._full_nodeIds[node[UIDKEY]] = self._full_num_nodes - 1
        self._full_nodeNames[node[NAMEKEY]] = self._full_num_nodes - 1
        
        self._full_edges_matrix = {k: self._add_null_line(v) for k,v in self._full_edges_matrix.items()}
        self._full_identity_matrix = self._add_null_line(self._full_identity_matrix)
        self._full_identity_matrix[self._full_num_nodes - 1, self._full_num_nodes - 1] = True
        return self._full_nodes, self._full_edges

    def fuse_node(self, delete_node, remain_uid):
        if isinstance(remain_uid, Dict):
            remain_uid = remain_uid[UIDKEY]
        _targ_uid = remain_uid
        _targ_nodeId = self._full_nodeIds[_targ_uid]
        remain_node = self._full_nodes[_targ_nodeId]
        if (not UIDKEY in delete_node.keys()) or (not delete_node[UIDKEY].replace(" ", "")):
            delete_node[UIDKEY] = get_uuid(name=delete_node[NAMEKEY])
        _src_uid = delete_node[UIDKEY]
        if _src_uid in self._full_nodeIds.keys():
            _src_nodeId = self._full_nodeIds[_src_uid]
            delete_node = self._full_nodes.pop(_src_nodeId)
            _new_inds = [i for i in range(self._full_num_nodes + 1) if i != _src_nodeId]
            self._full_identity_matrix = self._full_identity_matrix[:, _new_inds][_new_inds, :]
            for k, v in self._full_edges_matrix.items():
                add_row = v[_src_nodeId, :]
                add_col = v[:, _src_nodeId] #.copy()
                v[_targ_nodeId, :] += add_row
                v[:, _targ_nodeId] += add_col
                self._full_edges_matrix[k] = v[_new_inds, :][:, _new_inds] - self._full_identity_matrix
                
            ## Fuse Edges
            _edges = []
            for edge in self._full_edges:
                if _src_uid != edge[SRCKEY] and _src_uid != edge[TARGKEY]:
                    _edges.append(edge)
                elif _src_uid == edge[SRCKEY]:
                    edge_targ_uid = edge[TARGKEY]
                    if edge_targ_uid != _targ_uid:
                        edge_ = self._fuse_edge_properties({**edge, SRCKEY: _targ_uid}, edge, dirn=self._dir_path)
                        edgeRef = f"SRC@{edge_[SRCKEY]}-TARG@{edge_[TARGKEY]}-TYPE@{edge_.get(CONNTYPKEY, 'unlabeled')}"
                        if edgeRef not in self._full_edgesIds.keys():
                            _edges.append(edge_)
                    if (not self._dir_path is None):
                        self._del_edge_properties(edge, dirn=self._dir_path)
                else:
                    edge_src_uid = edge[SRCKEY]
                    if edge_src_uid != _targ_uid:
                        edge_ = self._fuse_edge_properties({**edge, TARGKEY: _targ_uid}, edge, dirn=self._dir_path)
                        edgeRef = f"SRC@{edge_[SRCKEY]}-TARG@{edge_[TARGKEY]}-TYPE@{edge_.get(CONNTYPKEY, 'unlabeled')}"
                        if edgeRef not in self._full_edgesIds.keys():
                            _edges.append(edge_)
                    if (not self._dir_path is None):
                        self._del_edge_properties(edge, dirn=self._dir_path)
        
            self._full_edges = _edges
            self._remap_nodes()

        _targ_nodeId = self._full_nodeIds[_targ_uid]
        
        if (not self._dir_path is None):
            node = self._fuse_node_properties(remain_node, delete_node, dirn=self._dir_path)
        else:
            node = {**remain_node, NODEPROPKEY: {**delete_node.get(NODEPROPKEY, {}),
                                                 **remain_node.get(NODEPROPKEY, {})}}
        node[UIDKEY] = _targ_uid
            
        if _src_uid in self._full_nodeIds.keys() and (not self._dir_path is None):
            self._del_node_properties(delete_node, dirn=self._dir_path)
                
        self._full_nodes[_targ_nodeId] = node      
        
        self._remap_edges()  
        
        #print(self._full_edges, _targ_uid, _src_uid)        
        
        return self._full_nodes, self._full_edges

    def del_node(self, node):
        _uid = node[UIDKEY]
        _nodeId = self._full_nodeIds[_uid]
        self._full_nodes.pop(_nodeId)

        _edges = []
        for edge in self._full_edges:
            if _uid != edge[SRCKEY] and _uid != edge[TARGKEY]:
                _edges.append(edge)
            elif (not self._dir_path is None):
                self._del_edge_properties(edge, dirn=self._dir_path)
                
        self._full_edges = _edges
        
        _n_inds = [i for i in range(self._full_num_nodes + 1) if i != _nodeId]
        self._full_identity_matrix = self._full_identity_matrix[:, _n_inds][_n_inds, :]
        self._full_edges_matrix = {
            k: v[:, _n_inds][_n_inds, :] for k, v in self._full_edges_matrix.items()}

        if (not self._dir_path is None):
            self._del_node_properties(node, dirn=self._dir_path)
        
        self._remap_nodes()
        self._remap_edges()
        
        return self._full_nodes, self._full_edges

    def edit_node(self, node):
        _uid = node[UIDKEY]
        _nodeId = self._full_nodeIds[_uid]

        if (not self._dir_path is None):
            node = self._fuse_node_properties(self._full_nodes[_nodeId], node, dirn=self._dir_path)
                
        self._full_nodes[_nodeId] = node

        return self._full_nodes, self._full_edges

    def get_node_info(self, node_uid):
        _nodeId = self._full_nodeIds[node_uid]
        node = self._full_nodes[_nodeId]
        return self._read_node_properties(node, dirn=self._dir_path)

    def get_edge_info(self, srcUid, trgUid, conn_type):
        conn_type = conn_type or 'unlabeled'
        edgeRef = f"SRC@{srcUid}-TARG@{trgUid}-TYPE@{conn_type}"
        edge = self._full_edges[self._full_edgesIds[edgeRef]]
        return self._read_edge_properties(edge, dirn=self._dir_path)
        
    def save(self):
        with open(os.path.join(self._dir_path, "structJson.gmdb"), "w+", encoding="utf-8") as f:
            json.dump(self.struct, f, indent=4, ensure_ascii=False)

    def load(self):
        if os.path.exists(os.path.join(self._dir_path, "structJson.gmdb")):
            with open(os.path.join(self._dir_path, "structJson.gmdb"), "r+", encoding="utf-8") as f:
                struct = json.load(f)
                self.restart(nodes=struct.get("nodes", []), edges=struct.get("edges", []))
            
if __name__ == "__main__":
    gm = GraphManage(
        nodes=[],
        edges=[]
        )
    
    gm.add_node({"name": "电阻", NODETYPKEY: "元件"})
    gm.add_node({"name": "电阻_1", NODETYPKEY: "元件"})
    gm.add_node({"name": "短路", NODETYPKEY: "故障"})
    gm.add_edge({CONNTYPKEY: "导致", SRCKEY: gm._full_nodes[0][UIDKEY], TARGKEY: gm._full_nodes[2][UIDKEY]})
    gm.add_edge({CONNTYPKEY: "导致", SRCKEY: gm._full_nodes[1][UIDKEY], TARGKEY: gm._full_nodes[2][UIDKEY]})
    gm.fuse_node(gm._full_nodes[1], gm._full_nodes[0][UIDKEY])
    gm.del_node(gm._full_nodes[0])
    print(gm._full_nodes, gm._full_edges)
