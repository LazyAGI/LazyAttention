import unittest
from lazy_attention.planning import encode_rows


def decode(data, size):
    blocks = size // 64
    result = []
    for block in range(blocks):
        head, ptr = data[block], data[blocks+block]
        if head >= 0:
            result.extend(range(head,head+64))
        elif head == -64:
            result.extend(data[ptr:ptr+64])
        else:
            cut = -head
            result.extend(range(data[ptr],data[ptr]+cut))
            result.extend(range(data[ptr+1],data[ptr+1]+64-cut))
    return result


class PlanningTests(unittest.TestCase):
    def test_all_splits_without_index_payload(self):
        for cut in range(1,64):
            index = list(range(cut,64)) + list(range(cut))
            data,info = encode_rows(index)
            self.assertEqual(decode(data,64),index)
            self.assertEqual(info['mode_counts'],[0,1,0])
            self.assertEqual(len(data),4)  # two headers + two bases
    def test_mixed(self):
        index = list(range(64)) + list(range(65,128))+[64]+list(range(191,127,-1))
        data,info = encode_rows(index)
        self.assertEqual(decode(data,192),index)
        self.assertEqual(info['mode_counts'],[1,1,1])
        self.assertEqual(len(data),6+2+64)
    def test_reject_invalid(self):
        for index in ([],list(range(63)),[0]*64,list(range(63))+[64],[float(x) for x in range(64)]):
            with self.assertRaises(ValueError): encode_rows(index)

if __name__ == '__main__': unittest.main()
