# !/usr/bin/python -u

#
# Protocol Informatics Prototype
# Written by Marshall Beddoe <mbeddoe@baselineresearch.net>
# Copyright (c) 2004 Baseline Research
#
# Licensed under the LGPL
#

from PI import *
import os, sys, getopt
import pdb
import time


def main():
    print "Protocol Informatics Prototype (v0.01 beta)"
    print "Written by Marshall Beddoe <mbeddoe@baselineresearch.net>"
    print "Copyright (c) 2004 Baseline Research\n"

    # Defaults
    format = None
    weight = 1.0
    graph = False

    #
    # Parse command line options and do sanity checking on arguments
    #
    try:
        (opts, args) = getopt.getopt(sys.argv[1:], "pagw:")
    except:
        usage()

    for o, a in opts:
        if o in ["-p"]:
            format = "pcap"
        elif o in ["-a"]:
            format = "ascii"
        elif o in ["-w"]:
            weight = float(a)
        elif o in ["-g"]:
            graph = True
        else:
            usage()

    if len(args) == 0:
        usage()

    if weight < 0.0 or weight > 1.0:
        print "FATAL: Weight must be between 0 and 1"
        sys.exit(-1)

    # exit()
    # parameter:
    # input1 eq  filepath dir = "/home/monkey/PSA/DocumentTopic/smtp_per_97_control"
    # input2 eq cluster ###
    # file eq configuration file ------clusters

    input1 = sys.argv[len(sys.argv) - 2]
    clusterno = int(input1)
    print "clusterno:", clusterno

    path = sys.argv[len(sys.argv) - 3]
    file = sys.argv[len(sys.argv) - 4]
    save_file=sys.argv[len(sys.argv) - 5]
    th=sys.argv[-1]
    print path
    print file
    print th

    #
    # Open file and get sequences
    #
    start = time.clock()

    if format == "pcap":
        try:
            sequences = input.Pcap(file)
        except:
            print "FATAL: Error opening '%s'" % file
            sys.exit(-1)
    elif format == "ascii":
        try:
            # sequences = input.ASCII(file, filelist)
            print format
            sequences=input.ASCII2(path,file,clusterno)

            print sequences
        except:
            print "FATAL: Error opening '%s'" % file
            sys.exit(-1)
    else:
        print "FATAL: Specify file format"
        sys.exit(-1)

    if len(sequences) == 0:
        print "FATAL: No sequences found in '%s'" % file
        sys.exit(-1)
    else:
        print "Found %d unique sequences in '%s'" % (len(sequences), file)

    # exit()
    #
    # Create distance matrix (LocalAlignment, PairwiseIdentity, Entropic)
    #
    print "Creating distance matrix ..",
    dmx = distance.LocalAlignment(sequences)
    # dmx = distance.PairwiseIdentity(sequences)
    print "complete"

    # added by wyp
    # sys.exit(0)

    #
    # Pass distance matrix to phylogenetic creation function
    #
    print "Creating phylogenetic tree .."
    phylo = phylogeny.UPGMA(sequences, dmx, minval=weight)
    print "complete"

    #
    # Output some pretty graphs of each cluster
    #
    if graph:
        print('mmm')
        cnum = 1
        for cluster in phylo:
            out = "graph-%d" % cnum
            print "Creating %s .." % out,
            cluster.graph(out)
            print "complete"
            cnum += 1

    print('www')

    print "\nDiscovered %d clusters using a weight of %.02f" % (len(phylo), weight)

    #
    # Perform progressive multiple alignment against clusters
    #

    i = 1
    alist = []
    for cluster in phylo:
        print "Performing multiple alignment on cluster %d .." % i,
        aligned = multialign.NeedlemanWunsch(cluster)
        print "complete"
        alist.append(aligned)
        i += 1
    print ""

    # alist = []
    # aligned = multialign.NeedlemanWunsch(phylo)
    # print "complete"
    # alist.append(aligned)
    # print ""

    # added by wyp

    # sys.exit(0)
    elapsed = (time.clock() - start)
    #
    # Display each cluster of aligned sequences
    #
    print(clusterno)
    print alist
    # print(file)
    textname=save_file
    print(textname)

    i = 1
    for seqs in alist:
        print "Output of cluster %d" % i
        # pdb.set_trace()
        print seqs
        output.Ansi(seqs, clusterno,textname,th)


        i += 1
    print ""

    print("Time used:", elapsed)


if __name__ == "__main__":
    main()
