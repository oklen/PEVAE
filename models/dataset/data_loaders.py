from __future__ import print_function
import numpy as np
from torch._C import dtype
from models.utils import Pack
from models.dataset.dataloader_bases import DataLoader
from models.dataset.corpora import aNLG_feature
import torch


# Stanford Multi Domain
class SMDDataLoader(DataLoader):
    def __init__(self, name, data, config):
        super(SMDDataLoader, self).__init__(name, fix_batch=config.fix_batch)
        self.name = name
        self.max_utt_size = config.max_utt_len
        self.data = self.flatten_dialog(data, config.backward_size)
        self.data_size = len(self.data)
        if config.fix_batch:
            all_ctx_lens = [len(d.context) for d in self.data]
            self.indexes = list(np.argsort(all_ctx_lens))[::-1]
        else:
            self.indexes = list(range(len(self.data)))

    def flatten_dialog(self, data, backward_size):
        results = []
        for dialog in data:
            for i in range(1, len(dialog)):
                e_id = i
                s_id = max(0, e_id - backward_size)
                response = dialog[i].copy()
                # response['utt_orisent'] = response.utt
                response['utt'] = self.pad_to(self.max_utt_size, response.utt, do_pad=False)
                contexts = []
                for turn in dialog[s_id:e_id]:
                    turn['utt'] = self.pad_to(self.max_utt_size, turn.utt, do_pad=False)
                    contexts.append(turn)
                results.append(Pack(context=contexts, response=response))
        return results

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]
        # input_context, context_lens, floors, topics, a_profiles, b_Profiles, outputs, output_lens
        context_lens, context_utts, out_utts, out_lens = [], [], [], []
        metas = []
        for row in rows:
            ctx = row.context
            resp = row.response

            out_utt = resp.utt
            context_lens.append(len(ctx))
            context_utts.append([turn.utt for turn in ctx])

            out_utt = out_utt
            out_utts.append(out_utt)
            out_lens.append(len(out_utt))
            metas.append(resp.meta)
            # ori_out_utts.append(resp.utt_orisent)

        vec_context_lens = np.array(context_lens)
        vec_context = np.zeros((self.batch_size, np.max(vec_context_lens),
                                self.max_utt_size), dtype=np.int32)
        vec_outs = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
        vec_out_lens = np.array(out_lens)

        for b_id in range(self.batch_size):
            vec_outs[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
            # fill the context tensor
            new_array = np.empty((vec_context_lens[b_id], self.max_utt_size))
            new_array.fill(0)
            for i, row in enumerate(context_utts[b_id]):
                for j, ele in enumerate(row):
                    new_array[i, j] = ele
            vec_context[b_id, 0:vec_context_lens[b_id], :] = new_array

        return Pack(contexts=vec_context, context_lens=vec_context_lens,
                    outputs=vec_outs, output_lens=vec_out_lens,
                    metas=metas)

class SMDDialogSkipLoader(DataLoader):
    def __init__(self, name, data, config):
        super(SMDDialogSkipLoader, self).__init__(name, fix_batch=config.fix_batch)
        self.name = name
        self.max_utt_size = config.max_utt_len
        self.data = self.flatten_dialog(data, config.backward_size)
        self.data_size = len(self.data)
        if config.fix_batch:
            all_ctx_lens = [len(d.context) for d in self.data]
            self.indexes = list(np.argsort(all_ctx_lens))[::-1]
        else:
            self.indexes = list(range(len(self.data)))
        self.source_type = "context"

    def flatten_dialog(self, data, backward_size):
        results = []
        for dialog in data:
            for i in range(1, len(dialog)-1):
                e_id = i
                s_id = max(0, e_id - backward_size)

                response = dialog[i]
                prev = dialog[i - 1]
                next = dialog[i + 1]

                response['utt'] = self.pad_to(self.max_utt_size, response.utt, do_pad=False)
                prev['utt'] = self.pad_to(self.max_utt_size, prev.utt, do_pad=False)
                next['utt'] = self.pad_to(self.max_utt_size, next.utt, do_pad=False)

                contexts = []
                for turn in dialog[s_id:e_id]:
                    turn['utt'] = self.pad_to(self.max_utt_size, turn.utt, do_pad=False)
                    contexts.append(turn)

                results.append(Pack(context=contexts, response=response,
                                    prev_resp=prev, next_resp=next))
        return results

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]

        context_lens, context_utts, out_utts, out_lens = [], [], [], []
        prev_utts, prev_lens = [], []
        next_utts, next_lens = [], []
        metas = []
        for row in rows:
            ctx = row.context
            resp = row.response

            out_utt = resp.utt
            context_lens.append(len(ctx))
            context_utts.append([turn.utt for turn in ctx])

            out_utt = out_utt
            out_utts.append(out_utt)
            out_lens.append(len(out_utt))
            metas.append(resp.meta)

            prev_utts.append(row.prev_resp.utt)
            prev_lens.append(len(row.prev_resp.utt))

            next_utts.append(row.next_resp.utt)
            next_lens.append(len(row.next_resp.utt))

        vec_context_lens = np.array(context_lens)
        vec_context = np.zeros((self.batch_size, np.max(vec_context_lens),
                                self.max_utt_size), dtype=np.int32)
        vec_outs = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
        vec_prevs = np.zeros((self.batch_size, np.max(prev_lens)), dtype=np.int32)
        vec_nexts = np.zeros((self.batch_size, np.max(next_lens)),dtype=np.int32)
        vec_out_lens = np.array(out_lens)
        vec_prev_lens = np.array(prev_lens)
        vec_next_lens = np.array(next_lens)

        for b_id in range(self.batch_size):
            vec_outs[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
            vec_prevs[b_id, 0:vec_prev_lens[b_id]] = prev_utts[b_id]
            vec_nexts[b_id, 0:vec_next_lens[b_id]] = next_utts[b_id]

            # fill the context tensor
            new_array = np.empty((vec_context_lens[b_id], self.max_utt_size))
            new_array.fill(0)
            for i, row in enumerate(context_utts[b_id]):
                for j, ele in enumerate(row):
                    new_array[i, j] = ele
            vec_context[b_id, 0:vec_context_lens[b_id], :] = new_array

        return Pack(contexts=vec_context, context_lens=vec_context_lens,
                    outputs=vec_outs, output_lens=vec_out_lens,
                    metas=metas, prevs=vec_prevs, prev_lens=vec_prev_lens,
                    nexts=vec_nexts, next_lens=vec_next_lens)

# Daily Dialog
class DailyDialogSkipLoader(DataLoader):
    def __init__(self, name, data, config):
        super(DailyDialogSkipLoader, self).__init__(name, fix_batch=config.fix_batch)
        self.name = name
        self.max_utt_size = config.max_utt_len
        self.data = self.flatten_dialog(data, config.backward_size)
        self.data_size = len(self.data)
        if config.fix_batch:
            all_ctx_lens = [len(d.context) for d in self.data]
            self.indexes = list(np.argsort(all_ctx_lens))[::-1]
        else:
            self.indexes = list(range(len(self.data)))

    def flatten_dialog(self, data, backward_size):
        results = []
        for dialog in data:
            for i in range(1, len(dialog)-1):
                e_id = i
                s_id = max(0, e_id - backward_size)

                response = dialog[i]
                prev = dialog[i - 1]
                next = dialog[i + 1]

                response['utt'] = self.pad_to(self.max_utt_size,response.utt, do_pad=False)
                prev['utt'] = self.pad_to(self.max_utt_size, prev.utt, do_pad=False)
                next['utt'] = self.pad_to(self.max_utt_size, next.utt, do_pad=False)

                contexts = []
                for turn in dialog[s_id:e_id]:
                    turn['utt'] = self.pad_to(self.max_utt_size, turn.utt, do_pad=False)
                    contexts.append(turn)

                results.append(Pack(context=contexts, response=response,
                                    prev_resp=prev, next_resp=next))
        return results

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]

        context_lens, context_utts, out_utts, out_lens = [], [], [], []
        prev_utts, prev_lens = [], []
        next_utts, next_lens = [], []
        metas = []
        for row in rows:
            ctx = row.context
            resp = row.response

            out_utt = resp.utt
            context_lens.append(len(ctx))
            context_utts.append([turn.utt for turn in ctx])

            out_utt = out_utt
            out_utts.append(out_utt)
            out_lens.append(len(out_utt))
            metas.append(resp.meta)

            prev_utts.append(row.prev_resp.utt)
            prev_lens.append(len(row.prev_resp.utt))

            next_utts.append(row.next_resp.utt)
            next_lens.append(len(row.next_resp.utt))

        vec_context_lens = np.array(context_lens)
        vec_context = np.zeros((self.batch_size, np.max(vec_context_lens),
                                self.max_utt_size), dtype=np.int32)
        vec_outs = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
        vec_prevs = np.zeros((self.batch_size, np.max(prev_lens)), dtype=np.int32)
        vec_nexts = np.zeros((self.batch_size, np.max(next_lens)),dtype=np.int32)
        vec_out_lens = np.array(out_lens)
        vec_prev_lens = np.array(prev_lens)
        vec_next_lens = np.array(next_lens)

        for b_id in range(self.batch_size):
            vec_outs[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
            vec_prevs[b_id, 0:vec_prev_lens[b_id]] = prev_utts[b_id]
            vec_nexts[b_id, 0:vec_next_lens[b_id]] = next_utts[b_id]

            # fill the context tensor
            new_array = np.empty((vec_context_lens[b_id], self.max_utt_size))
            new_array.fill(0)
            for i, row in enumerate(context_utts[b_id]):
                for j, ele in enumerate(row):
                    new_array[i, j] = ele
            vec_context[b_id, 0:vec_context_lens[b_id], :] = new_array

        return Pack(contexts=vec_context, context_lens=vec_context_lens,
                    outputs=vec_outs, output_lens=vec_out_lens,
                    metas=metas, prevs=vec_prevs, prev_lens=vec_prev_lens,
                    nexts=vec_nexts, next_lens=vec_next_lens)

class DailyDialogSkipLoaderLabel(DataLoader):
    def __init__(self, name, data, config):
        super(DailyDialogSkipLoaderLabel, self).__init__(name, fix_batch=config.fix_batch)
        self.name = name
        self.max_utt_size = config.max_utt_len
        self.data = self.flatten_dialog(data, config.backward_size)
        self.data_size = len(self.data)
        if config.fix_batch:
            all_ctx_lens = [len(d.context) for d in self.data]
            self.indexes = list(np.argsort(all_ctx_lens))[::-1]
        else:
            self.indexes = list(range(len(self.data)))

    def flatten_dialog(self, data, backward_size):
        results = []
        for dialog in data:
            for i in range(1, len(dialog)-1):
                e_id = i
                s_id = max(0, e_id - backward_size)

                response = dialog[i]
                prev = dialog[i - 1]
                next = dialog[i + 1]

                response['utt'] = self.pad_to(self.max_utt_size,response.utt, do_pad=False)
                prev['utt'] = self.pad_to(self.max_utt_size, prev.utt, do_pad=False)
                next['utt'] = self.pad_to(self.max_utt_size, next.utt, do_pad=False)

                contexts = []
                for turn in dialog[s_id:e_id]:
                    turn['utt'] = self.pad_to(self.max_utt_size, turn.utt, do_pad=False)
                    contexts.append(turn)

                results.append(Pack(context=contexts, response=response,
                                    prev_resp=prev, next_resp=next))
        return results

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]

        context_lens, context_utts, out_utts, out_lens = [], [], [], []
        prev_utts, prev_lens = [], []
        next_utts, next_lens = [], []
        metas = []
        for row in rows:
            ctx = row.context
            resp = row.response

            out_utt = resp.utt
            context_lens.append(len(ctx))
            context_utts.append([turn.utt for turn in ctx])

            out_utt = out_utt
            out_utts.append(out_utt)
            out_lens.append(len(out_utt))
            metas.append(resp.meta)

            prev_utts.append(row.prev_resp.utt)
            prev_lens.append(len(row.prev_resp.utt))

            next_utts.append(row.next_resp.utt)
            next_lens.append(len(row.next_resp.utt))

        vec_context_lens = np.array(context_lens)
        vec_context = np.zeros((self.batch_size, np.max(vec_context_lens),
                                self.max_utt_size), dtype=np.int32)
        vec_outs = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
        vec_prevs = np.zeros((self.batch_size, np.max(prev_lens)), dtype=np.int32)
        vec_nexts = np.zeros((self.batch_size, np.max(next_lens)),dtype=np.int32)
        vec_out_lens = np.array(out_lens)
        vec_prev_lens = np.array(prev_lens)
        vec_next_lens = np.array(next_lens)

        for b_id in range(self.batch_size):
            vec_outs[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
            vec_prevs[b_id, 0:vec_prev_lens[b_id]] = prev_utts[b_id]
            vec_nexts[b_id, 0:vec_next_lens[b_id]] = next_utts[b_id]

            # fill the context tensor
            new_array = np.empty((vec_context_lens[b_id], self.max_utt_size))
            new_array.fill(0)
            for i, row in enumerate(context_utts[b_id]):
                for j, ele in enumerate(row):
                    new_array[i, j] = ele
            vec_context[b_id, 0:vec_context_lens[b_id], :] = new_array


        z_labels = np.zeros((self.batch_size, 2), dtype=np.int32)
        for b_id in range(self.batch_size):
            z_labels[b_id][0] = int(metas[b_id]["emotion"])
            z_labels[b_id][1] = int(metas[b_id]["act"])

        return Pack(contexts=vec_context, context_lens=vec_context_lens,
                    outputs=vec_outs, output_lens=vec_out_lens,
                    metas=metas, prevs=vec_prevs, prev_lens=vec_prev_lens,
                    nexts=vec_nexts, next_lens=vec_next_lens,
                    z_labels=z_labels)

class DailyDialogDataLoader(DataLoader):
    def __init__(self, name, data, config):
        super(DailyDialogDataLoader, self).__init__(name, fix_batch=config.fix_batch)
        self.name = name
        self.max_utt_size = config.max_utt_len
        self.data = self.flatten_dialog(data, config.backward_size)
        self.data_size = len(self.data)
        if config.fix_batch:
            all_ctx_lens = [len(d.context) for d in self.data]
            self.indexes = list(np.argsort(all_ctx_lens))[::-1]
        else:
            self.indexes = list(range(len(self.data)))

    def flatten_dialog(self, data, backward_size):
        results = []
        for dialog in data:
            for i in range(1, len(dialog)):
                e_id = i
                s_id = max(0, e_id - backward_size)
                response = dialog[i].copy()
                # response['utt_orisent'] = response.utt
                response['utt'] = self.pad_to(self.max_utt_size, response.utt, do_pad=False)
                contexts = []
                for turn in dialog[s_id:e_id]:
                    turn['utt'] = self.pad_to(self.max_utt_size, turn.utt, do_pad=False)
                    contexts.append(turn)
                results.append(Pack(context=contexts, response=response))
        return results

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]
        # input_context, context_lens, floors, topics, a_profiles, b_Profiles, outputs, output_lens
        context_lens, context_utts, out_utts, out_lens = [], [], [], []
        metas = []
        for row in rows:
            ctx = row.context
            resp = row.response

            out_utt = resp.utt
            context_lens.append(len(ctx))
            context_utts.append([turn.utt for turn in ctx])

            out_utt = out_utt
            out_utts.append(out_utt)
            out_lens.append(len(out_utt))
            metas.append(resp.meta)
            # ori_out_utts.append(resp.utt_orisent)

        vec_context_lens = np.array(context_lens)
        vec_context = np.zeros((self.batch_size, np.max(vec_context_lens),
                                self.max_utt_size), dtype=np.int32)
        vec_outs = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
        vec_out_lens = np.array(out_lens)

        for b_id in range(self.batch_size):
            vec_outs[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
            # fill the context tensor
            new_array = np.empty((vec_context_lens[b_id], self.max_utt_size))
            new_array.fill(0)
            for i, row in enumerate(context_utts[b_id]):
                for j, ele in enumerate(row):
                    new_array[i, j] = ele
            vec_context[b_id, 0:vec_context_lens[b_id], :] = new_array

        return Pack(contexts=vec_context, context_lens=vec_context_lens,
                    outputs=vec_outs, output_lens=vec_out_lens,
                    metas=metas)

# PTB
class PTBDataLoader(DataLoader):

    def __init__(self, name, data, config, max_utt_len=-1):
        super(PTBDataLoader, self).__init__(name, fix_batch=config.fix_batch)
        self.max_utt_size = config.max_utt_len if max_utt_len == -1 else max_utt_len
        self.data = self.pad_data(data)
        self.data_size = len(self.data)
        all_lens = [len(line.utt) for line in self.data]
        print("Max len %d and min len %d and avg len %f" % (np.max(all_lens),
                                                            np.min(all_lens),
                                                            float(np.mean(all_lens))))
        if config.fix_batch:
            self.indexes = list(np.argsort(all_lens))
        else:
            self.indexes = list(range(len(self.data)))

    def pad_data(self, data):
        for l in data:
            l['utt'] = self.pad_to(self.max_utt_size, l.utt, do_pad=False)
        return data

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]
        input_lens = np.array([len(row.utt) for row in rows], dtype=np.int32)
        max_len = np.max(input_lens)
        inputs = np.zeros((self.batch_size, max_len), dtype=np.int32)
        for idx, row in enumerate(rows):
            inputs[idx, 0:input_lens[idx]] = row.utt

        return Pack(outputs=inputs, output_lens=input_lens, metas=[data["meta"] for data in rows])

class aNLG_feature(object):
    def __init__(self,id,obs,seg_ids,input_masks,ans,ob1_end,obs_end,ans_end):
        self.obs_ids = obs
        self.obs_segment_ids = seg_ids
        self.input_masks = input_masks
        self.ans = ans
        self.ob1_end = ob1_end
        self.obs_end = obs_end
        self.ans_end = ans_end
        self.id = id

class ADRDataLoader(DataLoader):
    def __init__(self, name, data, config, max_utt_len=-1):
        self.data = data
        self.config = config
        self.name = name
        self.data_size = len(data)
        self.fix_batch = False
        self.indexes = list(range(len(self.data)))

    def pad_data(self,data): # We have pad it
        return data

    def _prepare_batch(self, selected_index):
        input_ids = []
        segment_ids = []
        input_mask = []
        input_len = []
        ans = []
        h1_end = []
        h2_end = []
        ans_end = []
        id = []

        for ids in selected_index:
            data_now = self.data[ids]
            if type(data_now) is list:
                three = data_now[0]
                input_ids.append(three[0])
                segment_ids.append(three[1])
                input_mask.append(three[2])
                o1_end = []

                for index in range(len(three[0])):
                    if index < data_now[1] and index > 0:
                        o1_end.append(1)
                    else:
                        o1_end.append(0)

                input_len.append(o1_end)

            else:
                id.append(data_now.id)
                input_ids.append(data_now.obs_ids)
                segment_ids.append(data_now.obs_segment_ids)
                input_mask.append(data_now.input_masks)
                ans.append(data_now.ans)
                o1_end = []
                o2_end = []
                ob1_end = data_now.ob1_end
                obs_end = data_now.obs_end

                for index in range(len(data_now.obs_ids)):
                    if index < ob1_end and index > 0:
                        o1_end.append(1)
                    else:
                        o1_end.append(0)

                    if index > ob1_end and index < obs_end:
                        o2_end.append(1)
                    else:
                        o2_end.append(0)

                h1_end.append(o1_end)
                h2_end.append(o2_end)
                ans_end.append(data_now.ans_end)
                
        if type(data_now) is list:
            return Pack(input_ids=torch.tensor(input_ids,dtype=torch.long).cuda(),\
                segment_ids=torch.tensor(segment_ids, dtype=torch.long).cuda()\
                    ,input_masks=torch.tensor(input_mask,dtype=torch.long).cuda(),
                    h1_end=torch.tensor(input_len,dtype=torch.long).cuda(),
                    h2_end=torch.tensor(input_len,dtype=torch.long).cuda(),
                    ans=torch.tensor(input_ids,dtype=torch.long).cuda())
        else:
            return Pack(id=torch.tensor(id,dtype=torch.long).cuda(),\
                input_ids=torch.tensor(input_ids,dtype=torch.long).cuda(),\
                    segment_ids=torch.tensor(segment_ids,dtype=torch.long).cuda(),\
                    input_masks=torch.tensor(input_mask,dtype=torch.long).cuda(),\
            ans=torch.tensor(ans,dtype=torch.long).cuda()\
                ,h1_end=torch.tensor(h1_end,dtype=torch.long).cuda()\
                    ,h2_end=torch.tensor(h2_end,dtype=torch.long).cuda(),
                    ans_end=torch.tensor(ans_end,dtype=torch.long).cuda())


class PWDataLoader(DataLoader):
    def __init__(self, name, data, config, max_utt_len=-1):
        self.data = data
        self.config = config
        self.name = name
        self.data_size = len(data)
        self.fix_batch = False
        self.indexes = list(range(len(self.data)))

    def pad_data(self,data): # We have pad it
        return data

    def _prepare_batch(self, selected_index):
        input_ids = []
        segment_ids = []
        input_mask = []
        input_ids1 = []
        segment_ids1 = []
        input_mask1 = []
        input_ids2 = []
        segment_ids2 = []
        input_mask2 = []
        ans = []
        cutLenght = self.config.max_seq_len

        for ids in selected_index:
            data_now = self.data[ids]
            if type(data_now.input_ids[0]) == list:
                input_ids1.append(data_now.input_ids[0])
                segment_ids1.append(data_now.segment_ids[0])
                input_mask1.append(data_now.input_mask[0])
                input_ids2.append(data_now.input_ids[1])
                segment_ids2.append(data_now.segment_ids[1])
                input_mask2.append(data_now.input_mask[1])
                ans.append(data_now.ans)
            else:
                input_ids.append(data_now.input_ids[:cutLenght])
                segment_ids.append(data_now.segment_ids[:cutLenght])
                input_mask.append(data_now.input_mask[:cutLenght])
                ans.append(data_now.input_ids[:cutLenght])

        if type(data_now.input_ids[0]) == list:
            return Pack(
                input_ids1=torch.tensor(input_ids1,dtype=torch.long).cuda(),\
                segment_ids1=torch.tensor(segment_ids1, dtype=torch.long).cuda()\
                    ,input_masks1=torch.tensor(input_mask1,dtype=torch.long).cuda(),
                input_ids2=torch.tensor(input_ids2,dtype=torch.long).cuda(),\
                segment_ids2=torch.tensor(segment_ids2, dtype=torch.long).cuda()\
                    ,input_masks2=torch.tensor(input_mask2,dtype=torch.long).cuda(),
                    ans=torch.tensor(ans,dtype=torch.long).cuda())
        else:
            return Pack(input_ids=torch.tensor(input_ids,dtype=torch.long).cuda(),\
                    segment_ids=torch.tensor(segment_ids,dtype=torch.long).cuda(),\
                    input_masks=torch.tensor(input_mask,dtype=torch.long).cuda(),\
            ans=torch.tensor(input_ids,dtype=torch.long).cuda())

import torch
class YELPDataLoader(DataLoader):
    def __init__(self, name, data, config, max_utt_len=-1):
        super(YELPDataLoader, self).__init__(name, fix_batch=config.fix_batch)
        self.config = config
        self.name = name
        self.fix_batch = False
        # self.data = self.pad_data(data)
        self.data = data
        self.indexes = list(range(len(self.data)))
        self.max_utt_size = config.max_seq_len
        self.data_size = len(data)

        # self.embedding = None
        self.rev_vocab = None
        self.sampler = None
        self.dataloader = None
        print("{} Data Count:{}".format(name,self.data_size))

    def pad_data(self, data):
        for l in data:
            l[-2] = self.pad_to(self.config.max_seq_len, l[-2])
        return data

    class batcher(object):
        def __init__(self,rev_vocab, max_seq_len, device):
            self.rev_vocab = rev_vocab
            self.max_seq_len = max_seq_len
            self.device = device
        @staticmethod
        def pad_to(max_len, tokens, do_pad=True):
            if len(tokens) >= max_len:
                return tokens[0:max_len - 1] + [tokens[-1]]
            elif do_pad:
                return tokens + [0] * (max_len - len(tokens))
            else:
                return tokens
        def __call__(self, rows):
            user_ids = []
            item_ids = [] 
            feature_ids = []
            sentence_embeddings = []
            input_ids = []
            ratings = []
            ans_ids = []
            rd_ids = []
            at_masks = []
            ui_ids = []
            pad_length = self.max_seq_len
            # pad_length = 0
            # for idx, row in enumerate(rows):
            #     pad_length = max(pad_length, len(row[3]) - 1)
            # pad_length = min(pad_length, self.max_seq_len)

            for idx, row in enumerate(rows):
                user_ids.append(rows[idx][0])
                item_ids.append(rows[idx][1])
                ui_ids.append(rows[idx][0] + 1000000000 * rows[idx][1])
                feature_ids.append(rows[idx][2])
                ratings.append(rows[idx][4]-1)
                input_id_now = []
                attention_mask = []

                for word in rows[idx][3]:
                    try:
                        input_id_now.append(self.rev_vocab[word])
                    except:
                        input_id_now.append(self.rev_vocab['<unk>'])
                    attention_mask.append(1)
                        
                # while len(attention_mask) <  pad_length:
                #     attention_mask.append(0)
                    
                ans_id = self.pad_to(pad_length, input_id_now[1:])
                # input_id_now = self.pad_to(pad_length, input_id_now[:-1])
                input_id_now = self.pad_to(pad_length, input_id_now)
                attention_mask = self.pad_to(pad_length, attention_mask[:-1])

                input_ids.append(input_id_now)
                ans_ids.append(ans_id)
                rd_ids.append(rows[idx][5])
                at_masks.append(attention_mask)

                if rows[idx][4] <= 0 or rows[idx][4] > 5:
                    print("ILLEGAL RATING!")
                    exit(0)

                # if self.embedding is not None:
                #     sentence_embeddings.append(self.embedding(torch.tensor(input_id_now,dtype=torch.long).cuda()))
                # else:
                sentence_embeddings.append(input_id_now)
                
            return {
                'user_ids':torch.tensor(user_ids, dtype=torch.long).to(self.device).contiguous(),
                'item_ids':torch.tensor(item_ids, dtype=torch.long).to(self.device).contiguous(),
                'feature_ids':torch.tensor(feature_ids, dtype=torch.long).to(self.device).contiguous(),
                'sentence_embeddings':torch.tensor(sentence_embeddings, dtype=torch.long).to(self.device).contiguous(), #Embedding transform already in GPU
                'attention_masks':torch.tensor(at_masks, dtype=torch.long).to(self.device).contiguous(),
                'ratings':torch.tensor(ratings, dtype=torch.float32).to(self.device).contiguous(),
                'input_ids':torch.tensor(input_ids, dtype=torch.long).to(self.device).contiguous(),
                'ans_ids':torch.tensor(ans_ids, dtype=torch.long).to(self.device).contiguous(),
                'rd_ids':torch.tensor(rd_ids, dtype=torch.float).to(self.device).contiguous(),
                'ui_ids':torch.tensor(ui_ids, dtype=torch.long).to(self.device).contiguous(),
            }

            # return Pack(
            #     user_ids = torch.tensor(user_ids, dtype=torch.long).cuda(),
            #     item_ids = torch.tensor(item_ids, dtype=torch.long).cuda(),
            #     feature_ids = torch.tensor(feature_ids, dtype=torch.long).cuda(),
            #     sentence_embeddings = torch.tensor(sentence_embeddings, dtype=torch.long).cuda(), #Embedding transform already in GPU
            #     ratings = torch.tensor(ratings, dtype=torch.long).cuda(),
            #     input_ids = torch.tensor(input_ids, dtype=torch.long).cuda(),
            # )

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]
        user_ids = []
        item_ids = [] 
        feature_ids = []
        sentence_embeddings = []
        input_ids = []
        ratings = []

        for idx, row in enumerate(rows):
            user_ids.append(rows[idx][0])
            item_ids.append(rows[idx][1])
            feature_ids.append(rows[idx][2])
            ratings.append(rows[idx][4]-1)
            input_id_now = []
            for word in rows[idx][3]:
                try:
                    input_id_now.append(self.rev_vocab[word])
                except:
                    input_id_now.append(self.rev_vocab['<unk>'])
            ans_ids = self.pad_to(self.config.max_seq_len, input_id_now[1:])
            input_id_now = self.pad_to(self.config.max_seq_len, input_id_now[:-1])

            input_ids.append(input_id_now)

            if rows[idx][4] <= 0 or rows[idx][4] > 5:
                print("ILLEGAL RATING!")
                exit(0)

            if self.embedding is not None:
                sentence_embeddings.append(self.embedding(torch.tensor(input_id_now,dtype=torch.long).cuda()))
            else:
                sentence_embeddings.append(rows[idx][3])
                print('Not Get Embedding in DataLoader!')

        return Pack(
            user_ids = torch.tensor(user_ids, dtype=torch.long).cuda(),
            item_ids = torch.tensor(item_ids, dtype=torch.long).cuda(),
            feature_ids = torch.tensor(feature_ids, dtype=torch.long).cuda(),
            sentence_embeddings = torch.stack(sentence_embeddings), #Embedding transform already in GPU
            ratings = torch.tensor(ratings, dtype=torch.float32).cuda(),
            input_ids = torch.tensor(input_ids, dtype=torch.long).cuda(),
            ans_ids = torch.tensor(ans_id, dtype=torch.long).cuda(),
        )
