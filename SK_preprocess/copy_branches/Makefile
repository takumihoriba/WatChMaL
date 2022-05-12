FLAGS=-g3 -DNDEBUG -pipe -m64 -std=c++11 -Wshadow -Wall -W -Woverloaded-virtual -fPIC -Wno-deprecated-declarations -pthread -MMD -MP
INCS=-I$(ROOTSYS)/include -I.

OBJS=loweroot_dict.o loweroot.o tqrealroot_dict.o tqrealroot.o

LIBS=-L$(ROOTSYS)/lib -lCore -lRIO -lHist -lGraf -lGraf3d -lGpad -lTree -lMatrix -lNet -lThread -lMathCore -lCore -lm -ldl  -pthread   -rdynamic

loweroot_dict.o: loweroot.h lowerootLinkDef.h
	rootcint -f lowerootDict.cc -c -I. loweroot.h lowerootLinkDef.h
	g++ -c $(FLAGS) $(INCS) -c lowerootDict.cc
	mv lowerootDict.o loweroot_dict.o
	mv lowerootDict.d loweroot_dict.d
	rm lowerootDict.cc


tqrealroot_dict.o: tqrealroot.h tqrealrootLinkDef.h
	rootcint -f tqrealrootDict.cc -c -I. tqrealroot.h tqrealrootLinkDef.h
	g++ -c $(FLAGS) $(INCS) -c tqrealrootDict.cc
	mv tqrealrootDict.o tqrealroot_dict.o
	mv tqrealrootDict.d tqrealroot_dict.d
	rm tqrealrootDict.cc

loweroot.o: 	loweroot.h
	g++ -c $(FLAGS) $(INCS) loweroot.cc

tqrealroot.o: 	tqrealroot.h
	g++ -c $(FLAGS) $(INCS) tqrealroot.cc

copy_branches.o:  copy_branches.cc
	g++ -c $(FLAGS) $(INCS) copy_branches.cc

copy_branches: copy_branches.o $(OBJS)
	g++ -o copy_branches copy_branches.o $(OBJS) $(LIBS)
