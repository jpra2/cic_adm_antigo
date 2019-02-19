import numpy as np
from math import pi, sqrt
from pymoab import core, types, rng, topo_util, skinner
import time
import pyximport; pyximport.install()
from PyTrilinos import Epetra, AztecOO, EpetraExt  # , Amesos
import scipy.sparse as sp

class Restriction:
    name_primal_tag = 'PRIMAL_ID_'
    name_fine_to_primal_tag = ['FINE_TO_PRIMAL', '_CLASSIC']
    name_wirebasket_id_tag = 'WIREBASKET_ID_LV'
    name_nc_primal = 'NC_PRIMAL_'
    name_d2_tag = 'd2'
    name_l1_id = 'l1_ID'
    name_l2_id = 'l2_ID'
    name_l3_id = 'l3_ID'

    @staticmethod
    def get_or_nv1(mb, op, map_wirebasket, wirebasket_numbers):
        name_primal_tag_level = Restriction.name_primal_tag + str(1)
        name_fine_to_primal_tag_level = Restriction.name_fine_to_primal_tag[0] + str(1) + Restriction.name_fine_to_primal_tag[1]
        primal_tag = mb.tag_get_handle(name_primal_tag_level)
        fine_to_primal_tag = mb.tag_get_handle(name_fine_to_primal_tag_level)
        d2_tag = mb.tag_get_handle(Restriction.name_d2_tag)
        ni = wirebasket_numbers[0]
        nf = wirebasket_numbers[1]
        ne = wirebasket_numbers[2]
        nv = wirebasket_numbers[3]
        vertex_elems = rng.Range([item[0] for item in map_wirebasket.items() if item[1] >= ni+nf+ne])

        meshsets = mb.get_entities_by_type_and_tag(
            mb.get_root_set(), types.MBENTITYSET, np.array([primal_tag]),
            np.array([None]))

        map_meshset_in_nc = []
        name_nc_primal_tag = Restriction.name_nc_primal + str(1)
        nc_primal_tag = mb.tag_get_handle(name_nc_primal_tag, 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
        OR = sp.lil_matrix((op.shape[1], op.shape[0]))

        interns = []
        faces = []
        edges = []
        verts = []
        map_nc_in_dual = {}

        for m in meshsets:
            elems = mb.get_entities_by_handle(m)
            vertex = rng.intersect(elems, vertex_elems)
            if len(vertex) != 1:
                print('erro')
                import pdb; pdb.set_trace()
            vertex = vertex[0]
            gids = [map_wirebasket[v] for v in elems]
            gid = map_wirebasket[vertex]
            line_op = op[gid]
            indice = sp.find(line_op)
            if len(indice[1]) != 1:
                print('erro')
                import pdb; pdb.set_trace()
            col = indice[1][0]
            map_meshset_in_nc.append(col)
            ones = np.ones(len(elems))
            OR[col, gids] = ones
            val_d2 = list(set(mb.tag_get_data(d2_tag, elems, flat=True)))
            if len(val_d2) > 1:
                print('erro')
                import pdb; pdb.set_trace()
            val_d2 = val_d2[0]
            map_nc_in_dual[col] = val_d2
            if val_d2 == 0:
                interns.append(col)
            elif val_d2 == 1:
                faces.append(col)
            elif val_d2 == 2:
                edges.append(col)
            elif val_d2 == 3:
                verts.append(col)
            else:
                raise ValueError('Erro no valor de d2_tag')

        mb.tag_set_data(nc_primal_tag, meshsets, map_meshset_in_nc)
        map_meshset_in_nc = dict(zip(np.array(meshsets), map_meshset_in_nc))



        # ids_nv1 = sorted(map_meshset_in_nc)
        wirebasket_numbers_nv1 = [len(interns), len(faces), len(edges), len(verts)]
        wirebasket_ord = interns + faces + edges + verts
        map_nc_in_wirebasket_id = dict(zip(wirebasket_ord, range(len(wirebasket_ord))))
        #########################################################
        # debug
        # nni  = len(interns)
        # nnf = nni + len(faces)
        # nne = nnf + len(edges)
        # nnv = nne + len(verts)
        #
        # nc_vertex_elems = np.array(wirebasket_ord[nne:nnv], dtype=np.uint64)
        # meshsets_vertex_elems_nv1 = [mb.get_entities_by_type_and_tag(
        #     mb.get_root_set(), types.MBENTITYSET, np.array([nc_primal_tag]),
        #     np.array([i]))[0] for i in nc_vertex_elems]
        # meshsets_vertex_elems_nv1 = rng.Range(meshsets_vertex_elems_nv1)
        #
        # cont = 0
        # for m in meshsets_vertex_elems_nv1:
        #     nc = mb.tag_get_data(nc_primal_tag, m, flat=True)[0]
        #     elems = mb.get_entities_by_handle(m)
        #     val_d2 = np.unique(mb.tag_get_data(d2_tag, elems, flat=True))
        #     print(val_d2)
        #     print(nc)
        #     print(nc in nc_vertex_elems)
        #     print('\n')
        #     if val_d2[0] != 3:
        #         print('erro')
        #         import pdb; pdb.set_trace()
        #     if cont == 10:
        #         cont = 0
        #         import pdb; pdb.set_trace()
        #     cont+=1
        #
        # print('saiu or1')
        # import pdb; pdb.set_trace()
        ######################################################################



        # ids_wirebasket = np.arange(len(wirebasket_ord))
        # map_ids_nv1_in_wirebasket = dict(zip(wirebasket_ord, ids_wirebasket))

        return OR, wirebasket_ord, wirebasket_numbers_nv1, map_meshset_in_nc, map_nc_in_wirebasket_id

    @staticmethod
    def get_or_nv2(mb, op, wirebasket_ord, wirebasket_numbers, map_meshset_in_nc_1, map_nc_in_wirebasket_id):
        map_wirebasket = map_nc_in_wirebasket_id
        name_primal_tag_level = Restriction.name_primal_tag + str(2)
        name_fine_to_primal_tag_level = Restriction.name_fine_to_primal_tag[0] + str(2) + Restriction.name_fine_to_primal_tag[1]
        primal_tag = mb.tag_get_handle(name_primal_tag_level)
        fine_to_primal_tag = mb.tag_get_handle(name_fine_to_primal_tag_level)

        ni = wirebasket_numbers[0]
        nf = wirebasket_numbers[1]
        ne = wirebasket_numbers[2]
        nv = wirebasket_numbers[3]

        nni = ni
        nnf = nni + nf
        nne = nnf + ne
        nnv = nne + nv

        meshsets_nv2 = mb.get_entities_by_type_and_tag(
            mb.get_root_set(), types.MBENTITYSET, np.array([primal_tag]),
            np.array([None]))

        map_meshset_in_nc = []
        name_nc_primal_tag = Restriction.name_nc_primal + str(2)
        nc_primal_tag = mb.tag_get_handle(name_nc_primal_tag, 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
        OR2 = sp.lil_matrix((op.shape[1], op.shape[0]))

        for m in meshsets_nv2:
            childs = mb.get_child_meshsets(m)
            ncs = [map_meshset_in_nc_1[v] for v in childs]
            ncs_wirebasket = [map_wirebasket[v] for v in ncs]

            vertex = [nc for nc in ncs_wirebasket if nc >= nne]
            if len(vertex) != 1:
                print('erro')
                import pdb; pdb.set_trace()
            vertex = vertex[0]
            gids = ncs_wirebasket
            gid = vertex
            line_op = op[gid]
            indice = sp.find(line_op)
            if len(indice[1]) != 1:
                print('erro')
                import pdb; pdb.set_trace()
            col = indice[1][0]
            map_meshset_in_nc.append(col)
            ones = np.ones(len(ncs))
            OR2[col, gids] = ones
            # val_d2 = list(set(mb.tag_get_data(d2_tag, elems, flat=True)))
            # if len(val_d2) > 1:
            #     print('erro')
            #     import pdb; pdb.set_trace()
            # val_d2 = val_d2[0]
            # if val_d2 == 0:
            #     interns.append(col)
            # elif val_d2 == 1:
            #     faces.append(col)
            # elif val_d2 == 2:
            #     edges.append(col)
            # elif val_d2 == 3:
            #     verts.append(col)
            # else:
            #     raise ValueError('Erro no valor de d2_tag')

        mb.tag_set_data(nc_primal_tag, meshsets_nv2, map_meshset_in_nc)
        map_meshset_in_nc = dict(zip(np.array(meshsets_nv2), map_meshset_in_nc))

        return OR2, map_meshset_in_nc

    @staticmethod
    def get_OR_ADM(mb, all_volumes, gids, map_wire_lv0, map_meshset_in_nc_1, map_nc_in_wirebasket_id_1, map_meshset_in_nc_2):
        map_global = dict(zip(np.array(all_volumes), gids))
        nc_primal_tag_1 = mb.tag_get_handle('NC_PRIMAL_1')
        nc_primal_tag_2 = mb.tag_get_handle('NC_PRIMAL_2')
        primal_tag_1 = mb.tag_get_handle('PRIMAL_ID_1')
        primal_tag_2 = mb.tag_get_handle('PRIMAL_ID_2')
        fine_to_primal_tag_1 = mb.tag_get_handle('FINE_TO_PRIMAL1_CLASSIC')
        fine_to_primal_tag_2 = mb.tag_get_handle('FINE_TO_PRIMAL2_CLASSIC')
        lv2_id_tag = mb.tag_get_handle(Restriction.name_l2_id)
        lv2_ids = mb.tag_get_data(lv2_id_tag, all_volumes, flat=True)
        l3_tag = mb.tag_get_handle(Restriction.name_l3_id)
        elems_nv0 = mb.get_entities_by_type_and_tag(
            mb.get_root_set(), types.MBHEX, np.array([l3_tag]),
            np.array([0]))
        # OR_adm = sp.lil_matrix((len(lv2_ids), len(all_volumes)))

        gids_adm = mb.tag_get_data(lv2_id_tag, all_volumes, flat=True)

        max_ids_adm = len(set(gids_adm))
        sz = (max_ids_adm, len(all_volumes))
        OR_adm = sp.lil_matrix(sz)

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
                OR_adm[i, gid_elem] = 1.0
            elif level == 2 or level == 3:

                gids_elems = [map_wire_lv0[v] for v in elems]
                colsf = np.array(gids_elems).astype(np.int32)
                linesf = np.repeat(i, len(gids_elems)).astype(np.int32)
                valuesf = np.ones(len(gids_elems))
                OR_adm[linesf, colsf] = valuesf

            else:
                raise ValueError('erro no valor de level')




            # or_adm_tag = self.mb.tag_get_handle(
            #     "OR_ADM_{0}".format(i), 1, types.MB_TYPE_INTEGER, True,
            #     types.MB_TAG_SPARSE, default_value=0)
            # self.mb.tag_set_data(or_adm_tag, elems, np.ones(len(elems), dtype=np.int))

        # linesf = linesf.astype(np.int32)
        # colsf = colsf.astype(np.int32)
        # inds_or_adm = np.array([linesf, colsf, valuesf, sz])
        # np.save('inds_or_adm', inds_or_adm)
        sp.save_npz('or_adm', OR_adm.tocsc(copy=True))
        return OR_adm

    @staticmethod
    def get_or_adm_nv1(mb, all_volumes, map_wire_lv0):
        nc_primal_tag_1 = mb.tag_get_handle('NC_PRIMAL_1')
        lv1_id_tag = mb.tag_get_handle(Restriction.name_l1_id)
        level_tag = mb.tag_get_handle(Restriction.name_l3_id)

        elems_nv0 = mb.get_entities_by_type_and_tag(
            mb.get_root_set(), types.MBHEX, level_tag,
            np.array([1]))

        ids_adm_nv1_elems_nv0 = mb.tag_get_data(lv1_id_tag, elems_nv0, flat=True)
        ids_wirebasket_elems_nv0 = np.array([map_wire_lv0[v] for v in elems_nv0])
        ids_adm_lv1 = mb.tag_get_data(lv1_id_tag, all_volumes, flat=True)
        max_id_lv1 = ids_adm_lv1.max()
        n = len(all_volumes)

        OR_adm_nv1 = sp.lil_matrix((max_id_lv1+1, n))
        OR_adm_nv1[ids_adm_nv1_elems_nv0, ids_wirebasket_elems_nv0] = np.ones(len(ids_adm_nv1_elems_nv0))

        meshsets = mb.get_entities_by_type_and_tag(
            mb.get_root_set(), types.MBENTITYSET, np.array([nc_primal_tag_1]),
            np.array([None]))

        for meshset in meshsets:
            elems = mb.get_entities_by_handle(meshset)
            inter = rng.intersect(elems_nv0, elems)
            if len(inter) > 0:
                continue
            # nc = mb.tag_get_data(nc_primal_tag_1, meshset, flat=True)[0]
            line_or_adm = mb.tag_get_data(lv1_id_tag, elems, flat=True)
            cols_or_adm = np.array([map_wire_lv0[v] for v in elems])
            OR_adm_nv1[line_or_adm, cols_or_adm] = np.ones(len(cols_or_adm))


        return OR_adm_nv1

    @staticmethod
    def get_or_adm_nv2(mb, all_volumes):

        nc_primal_tag_1 = mb.tag_get_handle('NC_PRIMAL_1')
        nc_primal_tag_2 = mb.tag_get_handle('NC_PRIMAL_2')
        fine_to_primal_tag_2 = mb.tag_get_handle('FINE_TO_PRIMAL2_CLASSIC')
        primal_tag_2 = mb.tag_get_handle('PRIMAL_ID_2')
        lv1_id_tag = mb.tag_get_handle(Restriction.name_l1_id)
        lv2_id_tag = mb.tag_get_handle(Restriction.name_l2_id)
        level_tag = mb.tag_get_handle(Restriction.name_l3_id)

        all_ids_nv1 = np.unique(mb.tag_get_data(lv1_id_tag, all_volumes, flat = True))
        all_ids_nv2 = np.unique(mb.tag_get_data(lv2_id_tag, all_volumes, flat = True))

        OR_adm_nv2 = sp.lil_matrix((len(all_ids_nv2), len(all_ids_nv1)))

        elems_nv0 = mb.get_entities_by_type_and_tag(
            mb.get_root_set(), types.MBHEX, np.array([level_tag]),
            np.array([1]))

        ids_adm_lv1 = mb.tag_get_data(lv1_id_tag, elems_nv0, flat=True)
        ids_adm_lv2 = mb.tag_get_data(lv2_id_tag, elems_nv0, flat=True)
        OR_adm_nv2[ids_adm_lv2, ids_adm_lv1] = np.ones(len(ids_adm_lv1))

        del elems_nv0

        elems_nv1 = mb.get_entities_by_type_and_tag(
            mb.get_root_set(), types.MBHEX, np.array([level_tag]),
            np.array([2]))

        ms = set()
        ids_adm_lv1 = []
        ids_adm_lv2 = []

        for elem in elems_nv1:
            if elem in ms:
                continue
            id_lv1 = mb.tag_get_data(lv1_id_tag, elem, flat=True)[0]
            elems = mb.get_entities_by_type_and_tag(
                mb.get_root_set(), types.MBHEX, np.array([lv1_id_tag]),
                np.array([id_lv1]))
            ms = ms | set(list(elems))
            id_lv2 = mb.tag_get_data(lv2_id_tag, elems, flat=True)[0]
            ids_adm_lv1.append(id_lv1)
            ids_adm_lv2.append(id_lv2)

        OR_adm_nv2[ids_adm_lv2, ids_adm_lv1] = np.ones(len(ids_adm_lv2))

        del ms
        del elems_nv1

        elems_nv2 = mb.get_entities_by_type_and_tag(
            mb.get_root_set(), types.MBHEX, np.array([level_tag]),
            np.array([3]))

        ms2 = set()
        ids_adm_lv1 = np.array([])
        ids_adm_lv2 = np.array([])

        for elem in elems_nv2:
            if elem in ms2:
                continue
            fine_to_primal_2 = mb.tag_get_data(fine_to_primal_tag_2, elem, flat=True)[0]
            meshset_lv2 = mb.get_entities_by_type_and_tag(
                mb.get_root_set(), types.MBENTITYSET, np.array([primal_tag_2]),
                np.array([fine_to_primal_2]))

            elems_2 = mb.get_entities_by_handle(meshset_lv2[0])
            id_adm_2 = np.unique(mb.tag_get_data(lv2_id_tag, elems_2, flat=True))
            if len(id_adm_2) > 1:
                print('erro')
                import pdb; pdb.set_trace()
            id_adm_2 = id_adm_2[0]
            ms2 = ms2 | set(list(elems_2))

            childs = mb.get_child_meshsets(meshset_lv2[0])

            for m in childs:
                elems_1 = mb.get_entities_by_handle(m)
                id_adm_1 = np.unique(mb.tag_get_data(lv1_id_tag, elems_1, flat=True))
                if len(id_adm_1) > 1:
                    print('erro')
                    import pdb; pdb.set_trace()
                # id_adm_1 = id_adm_1
                ids_adm_lv1 = np.append(ids_adm_lv1, id_adm_1)
            ids_adm_lv2 = np.append(ids_adm_lv2, np.repeat(id_adm_2, len(childs)))

        ids_adm_lv1 = ids_adm_lv1.astype(np.int32)
        ids_adm_lv2 = ids_adm_lv2.astype(np.int32)
        OR_adm_nv2[ids_adm_lv2, ids_adm_lv1] = np.ones(len(ids_adm_lv1))

        return OR_adm_nv2
