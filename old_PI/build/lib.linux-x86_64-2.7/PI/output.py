#pdb.set_trace()
"""
Consensus module
Generate consensus based on multiple sequence alignment

Written by Marshall Beddoe <mbeddoe@baselineresearch.net>
Copyright (c) 2004 Baseline Research

Licensed under the LGPL
"""

from curses.ascii import *
import pdb

class Output:

    def __init__(self, sequences, index,textname,th):

        self.sequences = sequences
        self.index = index
        self.consensus = []
        self.textname=textname
        self.th=th
        self._go()

    def _go(self):
        pass

class Ansi(Output):

    def __init__(self, sequences, index,textname,th):

        # Color defaults for composition
        self.gap = "\033[41;30m%s\033[0m"
        self.printable = "\033[42;30m%s\033[0m"
        self.space = "\033[43;30m%s\033[0m"
        self.binary = "\033[44;30m%s\033[0m"
        self.zero = "\033[45;30m%s\033[0m"
        self.bit = "\033[46;30m%s\033[0m"
        self.default = "\033[47;30m%s\033[0m"

        Output.__init__(self, sequences, index,textname,th)

    def _go(self):

        seqLength = len(self.sequences[0][1])
        rounds = seqLength / 18
        remainder = seqLength % 18
        l = len(self.sequences[0][1])

        start = 0
        end = 18

        dtConsensus = []
        mtConsensus = []

        for i in range(rounds):
            for id, seq in self.sequences:
                print "%04d" % id,
                for byte in seq[start:end]:
                    if byte == 256:
                        print self.gap % "___",
                    elif isspace(byte):
                        print self.space % "   ",
                    elif isprint(byte):
                        print self.printable % "x%02x" % byte,
                    elif byte == 0:
                        print self.zero % "x00",
                    else:
                        print self.default % "x%02x" % byte,
                print ""

            # Calculate datatype consensus


            print "DT  ",

            for j in range(start, end):
                column = []
                for id, seq in self.sequences:
                    column.append(seq[j])
                dt = self._dtConsensus(column)
                print dt,
                dtConsensus.append(dt)
            print ""

            print "MT  ",
            for j in range(start, end):
                column = []
                for id, seq in self.sequences:
                    column.append(seq[j])
                rate = self._mutationRate(column)
                print "%03d" % (rate * 100),
                mtConsensus.append(rate)
            print "\n"

            start += 18
            end += 18
            
   
        if remainder:
            for id, seq in self.sequences:
                print "%04d" % id,
                for byte in seq[start:start + remainder]:
                    if byte == 256:
                        print self.gap % "___",
                    elif isspace(byte):
                        print self.space % "   ",
                    elif isprint(byte):
                        print self.printable % "x%02x" % byte,
                    elif byte == 0:
                        print self.zero % "x00",
                    else:
                        print self.default % "x%02x" % byte,
                print ""

            print "DT  ",
            
            #pdb.set_trace()
            for j in range(start, start + remainder):
                column = []
                for id, seq in self.sequences:
                    column.append(seq[j])
                dt = self._dtConsensus(column)
                print dt,
                dtConsensus.append(dt)
            print ""

            print "MT  ",
            for j in range(start, start + remainder):
                column = []
                for id, seq in self.sequences:
                    column.append(seq[j])
                rate = self._mutationRate(column)
                mtConsensus.append(rate)
                print "%03d" % (rate * 100),
            print ""

        #pdb.set_trace()
        # Calculate consensus sequence
        l = len(self.sequences[0][1])

        for i in range(l):
            histogram = {}
            for id, seq in self.sequences:
                try:
                    histogram[seq[i]] += 1
                except:
                    histogram[seq[i]] = 1

            items = histogram.items()
            items.sort()

            m = 1
            v = 257
            for j in items:
                if j[1] > m:
                    m = j[1]
                    v = j[0]

            self.consensus.append(v)

            real = []

            for i in range(len(self.consensus)):
                #if self.consensus[i] == 256:
                #    continue
                real.append((self.consensus[i], dtConsensus[i], mtConsensus[i]))

        print(real)

        th=float(self.th)
        print(th)
        resultlis=[]
        for state in real:
            if(float(state[2])<th):
                byte=hex(state[0])[2:]
                if(len(byte)==1):
                    byte='0'+byte
                resultlis.append(byte)
            else:
                resultlis.append('??')
        print(resultlis)
        with open(self.textname,'a') as f:
            for byte in resultlis:
                f.write(byte+' ')
            f.write('\r\n')




        #
        # Display consensus data
        #
        #pdb.set_trace()    
        
        threshold = 20
        sequence = ''
        sequencelen = 0
        sign = 0
        counter = 0
        #random
        alpha = 0
        #fixed
        beta = 0


        totalLen = len(real)
        rounds = totalLen / 18
        remainder = totalLen % 18

        start = 0
        end = 18

        print "\nUngapped Consensus:"

        for i in range(rounds):

            print "CONS",
            for byte,type,rate in real[start:end]:
                if byte == 256:
                    print self.gap % "___",
                elif byte == 257:
                    print self.default % "???",
                elif isspace(byte):
                    print self.space % "   ",
                elif isprint(byte):
                    print self.printable % "x%02x" % byte,
                elif byte == 0:
                    print self.zero % "x00",
                else:
                    print self.default % "x%02x" % byte,


                #pdb.set_trace()

                if sequencelen < 50:
                    if byte == 256:
			if sequencelen == 0:
			    continue

			if sign == 0:
                            sequence = sequence + "([\\s\\S]*)"
                            sequencelen = sequencelen + 1
                            alpha = alpha + 1
                            sign = 1
                    elif rate*100 > threshold:
			if sequencelen == 0:
			    continue

                        if sign == 0:
                            sequence = sequence + "([\\s\\S]*)"
                            sequencelen = sequencelen + 1
                            alpha = alpha + 1
                            sign = 1
                    elif byte == 257:
			if sequencelen == 0:
			    continue

                        if sign == 0:
                            sequence = sequence + "([\\s\\S]*)"
                            sequencelen = sequencelen + 1
                            alpha = alpha + 1
                            sign = 1
                    else:
                        #pdb.set_trace()
                        sequence = sequence + "[\\\\x%02x]" % byte
                        sequencelen = sequencelen + 1
                        beta = beta + 1
                        sign = 0

            print ""


            print "DT  ",
            for byte,type,rate in real[start:end]:
                print type,
            print ""


            print "MT  ",
            for byte,type,rate in real[start:end]:
                print "%03d" % (rate * 100),
            print "\n"



            start += 18
            end += 18


        if remainder:

            print "CONS",
            for byte,type,rate in real[start:start + remainder]:
                if byte == 256:
                    print self.gap % "___",
                elif byte == 257:
                    print self.default % "???",
                elif isspace(byte):
                    print self.space % "   ",
                elif isprint(byte):
                    print self.printable % "x%02x" % byte,
                elif byte == 0:
                    print self.zero % "x00",
                else:
                    print self.default % "x%02x" % byte,



                if sequencelen < 50:
                    if byte == 256:
                        if sign == 0:
                            sequence = sequence + "([\\s\\S]*)"
                            sequencelen = sequencelen + 1
                            alpha = alpha + 1
                            sign = 1
                    elif rate*100 > threshold:
                        if sign == 0:
                            sequence = sequence + "([\\s\\S]*)"
                            sequencelen = sequencelen + 1
                            alpha = alpha + 1
                            sign = 1
                    elif byte == 257:
                        if sign == 0:
                            sequence = sequence + "([\\s\\S]*)"
                            sequencelen = sequencelen + 1
                            alpha = alpha + 1
                            sign = 1
                    else:
                        #pdb.set_trace()
                        sequence = sequence + "[\\\\x%02x]" % byte
                        sequencelen = sequencelen + 1
                        beta = beta + 1
                        sign = 0


            print ""

            print "DT  ",
            for byte,type,rate in real[start:end]:
                print type,
            print ""


            print "MT  ",
            for byte,type,rate in real[start:end]:
                print "%03d" % (rate * 100),
            print "\n"


        # #save the final results
        # if ((beta > alpha) and (beta > 3)):
        #     patStrFile = 'pattern%d' % self.index
        #     sequenceResult = open(patStrFile, 'w+')
        #     sequenceResult.write("\t%d:'" % self.index)
        #     sequenceResult.write(sequence)
        #     sequenceResult.write("',\n")
        #     sequenceResult.close()

    def _dtConsensus(self, data):
        histogram = {}

        for byte in data:
            if byte == 256:
                try:
                    histogram["G"] += 1
                except:
                    histogram["G"] = 1
            elif isspace(byte):
                try:
                    histogram["S"] += 1
                except:
                    histogram["S"] = 1
            elif isprint(byte):
                try:
                    histogram["A"] += 1
                except:
                    histogram["A"] = 1
            elif byte == 0:
                try:
                    histogram["Z"] += 1
                except:
                    histogram["Z"] = 1
            else:
                try:
                    histogram["B"] += 1
                except:
                    histogram["B"] = 1

        items = histogram.items()
        items.sort()

        m = 1
        v = '?'
        for j in items:
            if j[1] > m:
                m = j[1]
                v = j[0]

        return v * 3

    def _mutationRate(self, data):
	#pdb.set_trace()
        histogram = {}

        for x in data:
            try:
                histogram[x] += 1
            except:
                histogram[x] = 1

        items = histogram.items()
        items.sort()
         
	#pdb.set_trace()
	invariant = items[0][1]

	for i in range(1, len(items)):
	    if items[i][1] > invariant:
	        invariant = items[i][1]		
 
	if len(items) == 1:
            rate = 0.0
        else:
            rate = 1.0 - invariant * 1.0 / len(data) * 1.0
            #rate = len(items) * 1.0 / len(data) * 1.0
		

        return rate
