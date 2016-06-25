LIBS				=	-L /usr/local/lib/ -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_shape -lopencv_imgproc
LIBS				+= -lopencv_videoio -lpthread -lboost_filesystem -lboost_system -lopencv_objdetect
LIBS        +=  -lopencv_ml
INCLUDES		=	-I /usr/local/include -I .
CXXFLAGS		=	-std=c++1y
PROG				= test
SRC					= main.cc

${PROG}:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(PROG) $(LIBS) $(SRC)
clean:
	rm test
