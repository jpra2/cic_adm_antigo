import numpy as np
from math import pi, sqrt
import time
import pyximport; pyximport.install()
import math
import os
import shutil
import random
import sys
import io
import yaml
from pymoab import core, types, rng, topo_util, skinner
from trilinos_utils import TrilinosUtils as triutils
from others_utils import OtherUtils as oth
import scipy.sparse as sp
from scipy.sparse.linalg import inv



class ProlongationTPFA3D:
    name_primal_tag = 'PRIMAL_ID_'
    name_fine_to_primal_tag = ['FINE_TO_PRIMAL', '_CLASSIC']
    name_wirebasket_id_tag = 'WIREBASKET_ID_LV'
    name_nc_primal = 'NC_PRIMAL_'
    name_d2_tag = 'd2'
    name_l1_id = 'l1_ID'
    name_l2_id = 'l2_ID'
    name_l3_id = 'l3_ID'

    @staticmethod
    def get_tpfa_OP(comm, inds_mod, wirebasket_numbers):
        """
        obtem o operador de prolongamento wirebasket
        """

        ni = wirebasket_numbers[0]
        nf = wirebasket_numbers[1]
        ne = wirebasket_numbers[2]
        nv = wirebasket_numbers[3]

        idsi = ni
        idsf = idsi+nf
        idse = idsf+ne
        idsv = idse+nv
        loc = [idsi, idsf, idse, idsv]

        ntot = sum(wirebasket_numbers)

        OP = sp.lil_matrix((ntot, nv))
        t_mod = sp.lil_matrix((inds_mod[3][0], inds_mod[3][1]))
        t_mod[inds_mod[0], inds_mod[1]] = inds_mod[2]
        OP = ProlongationTPFA3D.insert_identity(OP, wirebasket_numbers)
        OP, inds_M = ProlongationTPFA3D.step1(comm, t_mod, OP, loc)
        OP, inds_M = ProlongationTPFA3D.step2(comm, t_mod, OP, loc, inds_M)
        OP = ProlongationTPFA3D.step3(comm, t_mod, OP, loc, inds_M)
        # OR = ProlongationTPFA3D.get_or(mb, OP)

        return OP

    @staticmethod
    def insert_identity(op, wirebasket_numbers):
        nv = wirebasket_numbers[3]
        nne = sum(wirebasket_numbers) - nv
        lines = np.arange(nne, nne+nv).astype(np.int32)
        values = np.ones(nv)
        matrix = sp.lil_matrix((nv, nv))
        rr = np.arange(nv).astype(np.int32)
        matrix[rr, rr] = values

        op[lines] = matrix

        return op

    @staticmethod
    def step1(comm, t_mod, op, loc):
        """
        elementos de aresta
        """
        lim = 1e-13

        nnf = loc[1]
        nne = loc[2]
        nnv = loc[3]
        ne = nne - nnf
        nv = nnv - nne
        nf = loc[1] - loc[0]
        M = t_mod[nnf:nne, nnf:nne]
        indices = M.nonzero()
        inds_M = [indices[0], indices[1], M[indices].toarray()[0], M.shape]
        inds_M = triutils.get_inverse_by_inds(comm, inds_M)
        M = triutils.get_CrsMatrix_by_inds(comm, inds_M)
        M2 = -1*t_mod[nnf:nne, nne:nnv]
        indices = M2.nonzero()
        inds_M2 = [indices[0], indices[1], M2[indices].toarray()[0], (ne, ne)]
        M2 = triutils.get_CrsMatrix_by_inds(comm, inds_M2)
        M = triutils.pymultimat(comm, M, M2)
        inds_M = triutils.get_inds_by_CrsMatrix(M)
        inds_M[3] = (ne, nv)
        matrix = sp.lil_matrix(inds_M[3])
        matrix[inds_M[0], inds_M[1]] = inds_M[2]

        op[nnf:nne] = matrix
        return op, inds_M

    @staticmethod
    def step2(comm, t_mod, op, loc, inds_MM):
        """
        elementos de face
        """
        nni = loc[0]
        nnf = loc[1]
        nne = loc[2]
        nnv = loc[3]
        ne = nne - nnf
        nv = nnv - nne
        nf = loc[1] - loc[0]
        ni = loc[0]

        if ne > nf:
            nt = ne
        else:
            nt = nf

        inds_MM[3] = (nt, nt)
        M = t_mod[nni:nnf, nni:nnf]
        # t0 = time.time()
        # Minv = inv(M.tocsc())
        # t1 = time.time()
        # print('tempo scipy')
        # print(t1-t0)
        # print('\n')
        indices = sp.find(M)
        inds_M = [indices[0], indices[1], indices[2], M.shape]
        # t0 = time.time()
        inds_M = triutils.get_inverse_by_inds(comm, inds_M)
        # t1 = time.time()
        # print('tempo trilinos')
        # print(t1-t0)
        # import pdb; pdb.set_trace()
        inds_M[3] = (nt, nt)
        M = triutils.get_CrsMatrix_by_inds(comm, inds_M) # nfxnf
        M2 = -1*t_mod[nni:nnf, nnf:nne] # nfxne
        indices = sp.find(M2)
        inds_M2 = [indices[0], indices[1], indices[2], (nt, nt)]
        M2 = triutils.get_CrsMatrix_by_inds(comm, inds_M2)
        M = triutils.pymultimat(comm, M, M2)
        MM = triutils.get_CrsMatrix_by_inds(comm, inds_MM)
        M = triutils.pymultimat(comm, M, MM)
        inds_M = triutils.get_inds_by_CrsMatrix(M)
        inds_M[3] = (nf, nv)
        matrix = sp.lil_matrix(inds_M[3])
        matrix[inds_M[0], inds_M[1]] = inds_M[2]

        op[nni:nnf] = matrix
        return op, inds_M

    @staticmethod
    def step3(comm, t_mod, op, loc, inds_MM):
        """
        elementos de face
        """
        nni = loc[0]
        nnf = loc[1]
        nne = loc[2]
        nnv = loc[3]
        ne = nne - nnf
        nv = nnv - nne
        nf = loc[1] - loc[0]
        ni = loc[0]

        nt = max([ni, nf, ne])

        inds_MM[3] = (nt, nt)
        M = t_mod[0:nni, 0:nni]
        indices = sp.find(M)
        inds_M = [indices[0], indices[1], indices[2], M.shape]
        inds_M = triutils.get_inverse_by_inds(comm, inds_M)
        inds_M[3] = (nt, nt)
        M = triutils.get_CrsMatrix_by_inds(comm, inds_M) # nfxnf
        M2 = -1*t_mod[0:nni, nni:nnf] # nfxne
        indices = sp.find(M2)
        inds_M2 = [indices[0], indices[1], indices[2], (nt, nt)]
        M2 = triutils.get_CrsMatrix_by_inds(comm, inds_M2)
        M = triutils.pymultimat(comm, M, M2)
        MM = triutils.get_CrsMatrix_by_inds(comm, inds_MM)
        M = triutils.pymultimat(comm, M, MM)
        inds_M = triutils.get_inds_by_CrsMatrix(M)
        inds_M[3] = (ni, nv)
        matrix = sp.lil_matrix(inds_M[3])
        matrix[inds_M[0], inds_M[1]] = inds_M[2]

        op[0:nni] = matrix
        return op

    @staticmethod
    def get_OP_adm(mb, all_volumes, gids, map_wire_lv0, map_meshset_in_nc_1, map_nc_in_wirebasket_id_1, map_meshset_in_nc_2, OP1, OP_n0_n2):


        map_global = dict(zip(np.array(all_volumes), gids))
        nc_primal_tag_1 = mb.tag_get_handle('NC_PRIMAL_1')
        nc_primal_tag_2 = mb.tag_get_handle('NC_PRIMAL_2')
        primal_tag_1 = mb.tag_get_handle('PRIMAL_ID_1')
        primal_tag_2 = mb.tag_get_handle('PRIMAL_ID_2')
        fine_to_primal_tag_1 = mb.tag_get_handle('FINE_TO_PRIMAL1_CLASSIC')
        fine_to_primal_tag_2 = mb.tag_get_handle('FINE_TO_PRIMAL2_CLASSIC')
        lv2_id_tag = mb.tag_get_handle(ProlongationTPFA3D.name_l2_id)
        lv2_ids = mb.tag_get_data(lv2_id_tag, all_volumes, flat=True)
        l3_tag = mb.tag_get_handle(ProlongationTPFA3D.name_l3_id)
        # elems_nv0 = mb.get_entities_by_type_and_tag(
        #     mb.get_root_set(), types.MBHEX, np.array([l3_tag]),
        #     np.array([0]))
        OP_adm = sp.lil_matrix((len(all_volumes), len(set(lv2_ids))))

        gids_adm = mb.tag_get_data(lv2_id_tag, all_volumes, flat=True)

        max_ids_adm = len(set(gids_adm))
        sz = (max_ids_adm, len(all_volumes))

        for i in range(max_ids_adm):

            elems = mb.get_entities_by_type_and_tag(
                mb.get_root_set(), types.MBHEX, lv2_id_tag,
                np.array([i]))
            # if len(elems) > 1:
            #     import pdb; pdb.set_trace()

            level = list(set(mb.tag_get_data(l3_tag, elems, flat=True)))
            if len(level) != 1:
                print('erro')
                import pdb; pdb.set_trace()
            level = level[0]
            if level == 1:
                gid_elem = map_wire_lv0[elems[0]]
                OP_adm[gid_elem, i] = 1.0
            elif level == 2:

                gids_elems = [map_wire_lv0[v] for v in elems]
                nc_classic = mb.tag_get_data(fine_to_primal_tag_1, elems, flat=True)[0]
                meshset = mb.get_entities_by_type_and_tag(
                    mb.get_root_set(), types.MBENTITYSET, np.array([primal_tag_1]),
                    np.array([nc_classic]))
                # nc = mb.tag_get_data(nc_primal_tag_1, meshset, flat=True)[0]
                nc = map_meshset_in_nc_1[list(meshset)[0]]
                nc2 = mb.tag_get_data(nc_primal_tag_1, meshset, flat=True)[0]
                # nc_wire = map_nc_in_wirebasket_id_1[nc]
                indices = sp.find(OP1[:, nc])
                cols = np.repeat(i, len(indices[0])).astype(np.int32)
                OP_adm[indices[0], cols] = indices[2]

            elif level == 3:

                gids_elems = [map_wire_lv0[v] for v in elems]
                nc_classic = mb.tag_get_data(fine_to_primal_tag_2, elems, flat=True)[0]
                meshset = mb.get_entities_by_type_and_tag(
                    mb.get_root_set(), types.MBENTITYSET, np.array([primal_tag_2]),
                    np.array([nc_classic]))
                # nc = mb.tag_get_data(nc_primal_tag_1, meshset, flat=True)[0]
                nc = map_meshset_in_nc_2[list(meshset)[0]]
                # nc2 = mb.tag_get_data(nc_primal_tag_1, meshset, flat=True)[0]
                # nc_wire = map_nc_in_wirebasket_id_1[nc]
                indices = sp.find(OP_n0_n2[:, nc])
                cols = np.repeat(i, len(indices[0])).astype(np.int32)
                OP_adm[indices[0], cols] = indices[2]

            else:
                raise ValueError('erro no valor de level prolonagtion')


        sp.save_npz('op_adm', OP_adm.tocsc(copy=True))
        return OP_adm

    @staticmethod
    def get_op_adm_nv1(mb, all_volumes, map_wire_lv0, OP1, vertex_elems):
        nc_primal_tag_1 = mb.tag_get_handle('NC_PRIMAL_1')
        lv1_id_tag = mb.tag_get_handle(ProlongationTPFA3D.name_l1_id)
        level_tag = mb.tag_get_handle(ProlongationTPFA3D.name_l3_id)
        vertex_elems = rng.Range(vertex_elems)

        elems_nv0 = mb.get_entities_by_type_and_tag(
            mb.get_root_set(), types.MBHEX, level_tag,
            np.array([1]))

        ids_adm_nv1_elems_nv0 = mb.tag_get_data(lv1_id_tag, elems_nv0, flat=True)
        ids_wirebasket_elems_nv0 = np.array([map_wire_lv0[v] for v in elems_nv0])
        ids_adm_lv1 = mb.tag_get_data(lv1_id_tag, all_volumes, flat=True)
        max_id_lv1 = ids_adm_lv1.max()
        n = len(all_volumes)

        OP_adm_nv1 = sp.lil_matrix((n, max_id_lv1+1))

        meshsets = mb.get_entities_by_type_and_tag(
            mb.get_root_set(), types.MBENTITYSET, np.array([nc_primal_tag_1]),
            np.array([None]))

        x = ids_wirebasket_elems_nv0
        for meshset in meshsets:
            elems = mb.get_entities_by_handle(meshset)
            vertex = rng.intersect(elems, vertex_elems)
            id_lv1_vertex = mb.tag_get_data(lv1_id_tag, vertex, flat=True)[0]
            # inter = rng.intersect(elems_nv0, elems)
            # # if len(inter) > 0:
            # #     continue
            nc = mb.tag_get_data(nc_primal_tag_1, meshset, flat=True)[0]
            # col_op = OP1[:,nc]
            # indices = sp.find(col_op)
            indices = sp.find(OP1[:,nc])
            y = indices[0]
            vals = indices[2]
            # xy = np.intersect1d(x, y)
            # if len(xy) > 0:
            #     retirar = []
            #     for i in xy:
            #         inds = np.where(y == i)[0][0]
            #         retirar.append(inds)
            #
            #     retirar = np.array(retirar)
            #     y = np.delete(y, retirar)
            #     vals = np.delete(vals, retirar)
            #
            # col_op_adm = mb.tag_get_data(lv1_id_tag, elems, flat=True)[0]
            col_op_adm = id_lv1_vertex
            col_op_adm = np.repeat(col_op_adm, len(y))
            OP_adm_nv1[y, col_op_adm] = vals

        for i in ids_wirebasket_elems_nv0:
            OP_adm_nv1[i] = np.zeros(max_id_lv1+1)

        OP_adm_nv1[ids_wirebasket_elems_nv0, ids_adm_nv1_elems_nv0] = np.ones(len(ids_adm_nv1_elems_nv0))

        return OP_adm_nv1

    @staticmethod
    def get_op_adm_nv2(mb, OP2, wirebasket_ord_1, wirebasket_numbers_1, map_nc_in_wirebasket_id_1, all_volumes):

        nc_primal_tag_1 = mb.tag_get_handle('NC_PRIMAL_1')
        nc_primal_tag_2 = mb.tag_get_handle('NC_PRIMAL_2')
        d2_tag = mb.tag_get_handle(ProlongationTPFA3D.name_d2_tag)

        map_wirebasket_id_in_nc_1 = {}
        for i, j in map_nc_in_wirebasket_id_1.items():
            map_wirebasket_id_in_nc_1[j] = i

        meshsets_nv2 = np.array(mb.get_entities_by_type_and_tag(
            mb.get_root_set(), types.MBENTITYSET, np.array([nc_primal_tag_2]),
            np.array([None])))

        lv1_id_tag = mb.tag_get_handle(ProlongationTPFA3D.name_l1_id)
        lv2_id_tag = mb.tag_get_handle(ProlongationTPFA3D.name_l2_id)
        level_tag = mb.tag_get_handle(ProlongationTPFA3D.name_l3_id)

        all_ids_nv1 = np.unique(mb.tag_get_data(lv1_id_tag, all_volumes, flat=True))
        all_ids_nv2 = np.unique(mb.tag_get_data(lv2_id_tag, all_volumes, flat=True))

        OP_adm_nv2 = sp.lil_matrix((len(all_ids_nv1), len(all_ids_nv2)))

        meshsets_nv1_que_estao_no_nv2 = []
        id_adm_lv2_dos_meshsets_nv1_que_estao_no_nv2 = []
        id_adm_lv1_dos_meshsets_nv1_que_estao_no_nv2 = []
        todos_meshsets_que_estao_no_nivel_1 = []
        ids_adm_nv1_de_todos_meshsets_que_estao_no_nivel_1 = []



        for m2 in meshsets_nv2:
            childs = mb.get_child_meshsets(m2)

            for m in childs:
                mb.add_parent_meshset(m, m2)
                elems = mb.get_entities_by_handle(m)
                id_adm_nv1 = np.unique(mb.tag_get_data(lv1_id_tag, elems, flat=True))
                if len(id_adm_nv1) > 1:
                    continue
                todos_meshsets_que_estao_no_nivel_1.append(m)
                ids_adm_nv1_de_todos_meshsets_que_estao_no_nivel_1.append(id_adm_nv1[0])

            elems = mb.get_entities_by_handle(m2)
            id_lv2 = np.unique(mb.tag_get_data(lv2_id_tag, elems, flat=True))
            if len(id_lv2) > 1:
                continue
            meshsets_nv1_que_estao_no_nv2.append(childs)
            id_adm_lv2_dos_meshsets_nv1_que_estao_no_nv2.append(id_lv2[0])
            ids_adm_lv1 = []
            for m in childs:
                elems_1 = mb.get_entities_by_handle(m)
                id_adm_lv1 = np.unique(mb.tag_get_data(lv1_id_tag, elems_1, flat=True))
                ids_adm_lv1.append(id_adm_lv1)
            id_adm_lv1_dos_meshsets_nv1_que_estao_no_nv2.append(ids_adm_lv1[:])

        ncs_de_todos_meshsets_que_estao_no_nivel_1 = mb.tag_get_data(nc_primal_tag_1, todos_meshsets_que_estao_no_nivel_1, flat=True)
        map_ncs_de_todos_meshsets_que_estao_no_nivel_1_in_id_adm_1 = dict(zip(ncs_de_todos_meshsets_que_estao_no_nivel_1, ids_adm_nv1_de_todos_meshsets_que_estao_no_nivel_1))
        map_todos_meshsets_que_estao_no_nivel_1_in_id_adm_1 = dict(zip(todos_meshsets_que_estao_no_nivel_1, ids_adm_nv1_de_todos_meshsets_que_estao_no_nivel_1))
        # import pdb; pdb.set_trace()
        # print(meshsets_nv1_que_estao_no_nv2)
        # print(id_adm_lv1_dos_meshsets_nv1_que_estao_no_nv2)
        # print(id_adm_lv2_dos_meshsets_nv1_que_estao_no_nv2)
        # print(todos_meshsets_que_estao_no_nivel_1)
        # print(ids_adm_nv1_de_todos_meshsets_que_estao_no_nivel_1)
        # import pdb; pdb.set_trace()

        nni = wirebasket_numbers_1[0]
        nnf = nni + wirebasket_numbers_1[1]
        nne = nnf + wirebasket_numbers_1[2]
        nnv = nne + wirebasket_numbers_1[3]

        nc_vertex_elems = np.array(wirebasket_ord_1[nne:nnv], dtype=np.uint64)
        meshsets_vertex_elems_nv1 = [mb.get_entities_by_type_and_tag(
            mb.get_root_set(), types.MBENTITYSET, np.array([nc_primal_tag_1]),
            np.array([i]))[0] for i in nc_vertex_elems]

        cont = 0
        vertices_pi_chapeu = []
        ids_lv2_vertices_pi_chapeu = []
        for m in meshsets_vertex_elems_nv1:
            if m in todos_meshsets_que_estao_no_nivel_1:
                vertices_pi_chapeu.append(m)
                elems = mb.get_entities_by_handle(m)
                id_lv2_adm = np.unique(mb.tag_get_data(lv2_id_tag, elems, flat=True))
                ids_lv2_vertices_pi_chapeu.append(id_lv2_adm)


        # meshsets_vertex_elems_nv1 = rng.Range(meshsets_vertex_elems_nv1)
        # nc_vertex_elems = mb.tag_get_data(nc_primal_tag_1, meshsets_vertex_elems_nv1, flat=True)

        for i, m in enumerate(vertices_pi_chapeu):
            id_adm_lv2_vertice = ids_lv2_vertices_pi_chapeu[i]
            parent_meshset = mb.get_parent_meshsets(m)
            nc2 = mb.tag_get_data(nc_primal_tag_2, parent_meshset, flat=True)[0]
            col_op2 = OP2[:,nc2]
            indices = sp.find(col_op2)
            lines = []
            vals = []
            for j, ind in enumerate(indices[0]):
                nc_1 = map_wirebasket_id_in_nc_1[ind]
                if nc_1 not in ncs_de_todos_meshsets_que_estao_no_nivel_1:
                    continue
                id_adm_nv1 = map_ncs_de_todos_meshsets_que_estao_no_nivel_1_in_id_adm_1[nc_1]
                lines.append(id_adm_nv1)
                vals.append(indices[2][j])
            col = np.repeat(id_adm_lv2_vertice, len(vals)).astype(np.int32)
            lines = np.array(lines).astype(np.int32)
            vals = np.array(vals)

            OP_adm_nv2[lines, col] = vals

        todos = rng.Range(todos_meshsets_que_estao_no_nivel_1)


        for meshsets in  meshsets_nv1_que_estao_no_nv2:
            todos = rng.subtract(todos, meshsets)

        for m in todos:
            elems = mb.get_entities_by_handle(m)
            id_adm_2 = np.unique(mb.tag_get_data(lv2_id_tag, elems, flat=True))[0]
            id_adm_1 = map_todos_meshsets_que_estao_no_nivel_1_in_id_adm_1[m]
            OP_adm_nv2[id_adm_1] = np.zeros(len(all_ids_nv2))
            OP_adm_nv2[id_adm_1, id_adm_2] = 1.0

        elems_nv0 = mb.get_entities_by_type_and_tag(
            mb.get_root_set(), types.MBHEX, np.array([level_tag]),
            np.array([1]))

        ids_adm_lv2_elems_nv0 = mb.tag_get_data(lv2_id_tag, elems_nv0, flat=True)
        ids_adm_lv1_elems_nv0 = mb.tag_get_data(lv1_id_tag, elems_nv0, flat=True)

        OP_adm_nv2[ids_adm_lv1_elems_nv0, ids_adm_lv2_elems_nv0] = np.ones(len(elems_nv0))

        return OP_adm_nv2
