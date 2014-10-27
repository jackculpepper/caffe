#define DLIB_JPEG_SUPPORT
#define DLIB_NO_GUI_SUPPORT
#define NO_MAKEFILE

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <iostream>

using namespace dlib;
using namespace std;


int main(int argc, char** argv) {  
  try {
    // This example takes in a shape model file and then a list of images to
    // process.  We will take these filenames in as command line arguments.
    // Dlib comes with example images in the examples/faces folder so give
    // those as arguments to this program.
    if (argc == 1) {
      cout << "Call this program like this:" << endl;
      cout << "./prepare_data.bin shape_predictor_68_face_landmarks.dat faces/*.jpg" << endl;
      cout << "You can get the shape_predictor_68_face_landmarks.dat file from:";
      cout << "http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2" << endl;
      return 0;
    }

    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor sp;
    deserialize(argv[1]) >> sp;

    // Loop over all the images provided on the command line.
    for (int i = 2; i < argc; ++i) {
      cout << "processing image " << argv[i] << endl;
      array2d<rgb_pixel> img;
      load_image(img, argv[i]);

      // Make the image larger so we can detect small faces.
      pyramid_up(img);

      // Detect
      std::vector<std::pair<double, rectangle> > dets;
      double adjust_threshold = -1.0;
      detector(img, dets, adjust_threshold);

      cout << "Number of faces detected: " << dets.size() << endl;

      // Align
      std::vector<full_object_detection> shapes;
      for (unsigned long j = 0; j < dets.size(); ++j) {
        cout << "det " << j << " ";
        cout << "score " << dets[j].first << " ";
        cout << "rect " << dets[j].second << endl;

        full_object_detection shape = sp(img, dets[j].second);

        cout << "num_parts " << shape.num_parts() << endl;

        for (int k = 0; k < shape.num_parts(); k++) {
          cout << "part " << k << " ";
          cout << "position " << shape.part(k) << endl;
        }
      }
    }
  } catch (exception& e) {
    cout << "\nexception thrown!" << endl;
    cout << e.what() << endl;
  }
}

