
"""
Input module

Handle different input file types and digitize sequences

Written by Marshall Beddoe <mbeddoe@baselineresearch.net>
Copyright (c) 2004 Baseline Research

Licensed under the LGPL
"""

from pcapy  import *
from socket import *
from sets   import *

__all__ = ["Input", "Pcap", "ASCII" ]

class Input:

    """Implementation of base input class"""

    def __init__(self,  filename):
        """Import specified filename"""

        self.set = Set()
        self.sequences = []
        self.index = 0

    def __iter__(self):
        self.index = 0
        return self

    def next(self):
        if self.index == len(self.sequences):
            raise StopIteration

        self.index += 1

        return self.sequences[self.index - 1]

    def __len__(self):
        return len(self.sequences)

    def __repr__(self):
        return "%s" % self.sequences

    def __getitem__(self, index):
        return self.sequences[index]

class Pcap(Input):

    """Handle the pcap file format"""

    def __init__(self, filename, offset=14):
        Input.__init__(self, filename)
        self.pktNumber = 0
        self.offset = offset

        try:
            pd = open_offline(filename)
        except:
            raise IOError

        pd.dispatch(-1, self.handler)

    def handler(self, hdr, pkt):
        if hdr.getlen() <= 0:
            return

        # Increment packet counter
        self.pktNumber += 1

        # Ethernet is a safe assumption
        offset = self.offset

        # Parse IP header
        iphdr = pkt[offset:]

        ip_hl = ord(iphdr[0]) & 0x0f                    # header length
        ip_len = (ord(iphdr[2]) << 8) | ord(iphdr[3])   # total length
        ip_p = ord(iphdr[9])                            # protocol type
        ip_srcip = inet_ntoa(iphdr[12:16])              # source ip address
        ip_dstip = inet_ntoa(iphdr[16:20])              # dest ip address

        offset += (ip_hl * 4)

        # Parse TCP if applicable
        if ip_p == 6:
            tcphdr = pkt[offset:]

            th_sport = (ord(tcphdr[0]) << 8) | ord(tcphdr[1])   # source port
            th_dport = (ord(tcphdr[2]) << 8) | ord(tcphdr[3])   # dest port
            th_off = ord(tcphdr[12]) >> 4                       # tcp offset

            offset += (th_off * 4)

        # Parse UDP if applicable
        elif ip_p == 17:
            offset += 8

        # Parse out application layer
        seq_len = (ip_len - offset) + 14

        if seq_len <= 0:
            return

        seq = pkt[offset:]

        l = len(self.set)
        self.set.add(seq)

        if len(self.set) == l:
            return

        # Digitize sequence
        digitalSeq = []
        for c in seq:
            digitalSeq.append(ord(c))

        self.sequences.append((self.pktNumber, digitalSeq))

class ASCII(Input):

    """Handle newline delimited ASCII input files"""

    def __init__(self, filename, filelist):
        Input.__init__(self, filename)

        lineno = 0

        for item in filelist:
            try:
                fd = open(item, "rb")
            except:
                raise IOError


            lineno += 1
            digitalSeq = []


            while 1:
                line = fd.readline()
                #print line

                if not line:
                    break

                # Digitize sequence
                for c in line:
                    digitalSeq.append(ord(c))

            #l = len(self.set)
            #self.set.add(tuple(digitalSeq))


            #if len(self.set) == l:
            #	continue

            #print digitalSeq
            self.sequences.append((lineno, digitalSeq))


def ASCII2(filename,clusters,clustersno):
    with open(filename, 'r') as f:
        datas = f.read().split('\n')
        datas = [x for x in datas if x != '']
        print('datas num:', len(datas))

    with open(clusters, 'r') as f:
        clusters_result = []
        result_length = []
        text = f.readlines()
        print('clusters num:', int(len(text) / 2))
        for line in text[1::2]:
            result = line.strip().split()
            clusters_result.append(result)
            result_length.append(len(result))
        # print(clusters_result)
        # print(result_length[:20])

    datas_cluster = []

    for cluster in clusters_result:
        dataslis = []
        for i in cluster:
            dataslis.append(datas[int(i)])
        datas_cluster.append(dataslis)

    sequences=[]
    lineno = 0
    for data in datas_cluster[clustersno-1]:
        digitalSeq = []
        lineno += 1
        # print(data)
        data_o=data.split()
        # print(data_o)
        for byte in data_o:
            # print(byte)
            digitalSeq.append(int(byte,16))

        sequences.append((lineno, digitalSeq))

    print clusters_result[clustersno-1]
    print lineno
    # fd=open(filename,'r')
    # digitalSeq = []
    # lineno = 0
    # while 1:
    #     lineno += 1
    #     line = fd.readline()
    #     # print line
    #
    #     if not line:
    #         break
    #
    #     # Digitize sequence
    #     for c in line:
    #         digitalSeq.append(ord(c))
    #     sequences.append((lineno, digitalSeq))
    return sequences

