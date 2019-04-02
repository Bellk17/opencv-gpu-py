#include <iostream>
#include <string>
#include <vector>

#include <Python.h>
#include "numpy/ndarrayobject.h"
#include "opencv_version.hpp"

#ifdef OPENCV_2
  #include <opencv2/core/gpumat.hpp> // gpu::getCudaEnabledDeviceCount
#else
  #include <opencv2/core/cuda.hpp>   // cuda::getCudaEnabledDeviceCount
#endif

#include <Python.h>

#include "face_detector.hpp"

static FaceDetector detector;
static bool init = false;


bool init_detector(PyObject* args, bool gpu)
{
  const char* cascade_file;
  if (init)
  {
    std::cout << "Warning:: Detector already exists" << std::endl;
  }
  else if (PyArg_ParseTuple(args, "s", &cascade_file))
  {
    detector.Init(std::string(cascade_file), gpu);
    init = true;
  }
  return false;
}

static PyObject* init_gpu_detector(PyObject* self, PyObject* args)
{
  if (init_detector(args, true))
  {
    Py_RETURN_TRUE;
  }
  else
  {
    Py_RETURN_FALSE;
  }
}

static PyObject* init_cpu_detector(PyObject* self, PyObject* args)
{
  if (init_detector(args, false))
  {
    Py_RETURN_TRUE;
  }
  else
  {
    Py_RETURN_FALSE;
  }
}

static PyObject* find_faces(PyObject* self, PyObject* args)
{
  if (init)
  {
    PyObject *size;
    PyArrayObject *image;

    if (PyArg_UnpackTuple(args, "ref", 1, 2, &size, &image))
    {
	    
      int rows = PyLong_AsLong(PyTuple_GetItem(size ,0));
      int cols = PyLong_AsLong(PyTuple_GetItem(size ,1));
      int nchannels = PyLong_AsLong(PyTuple_GetItem(size ,2));
	
      std::cout << "Rows: " << rows << std::endl;
      std::cout << "Cols: " << cols << std::endl;
      std::cout << "Channels: " << nchannels << std::endl;
      
      char my_arr[rows * nchannels * cols];

      for(size_t length = 0; length<(rows * nchannels * cols); length++){
	my_arr[length] = (*(char *)PyArray_GETPTR1(image, length));
      }

      cv::Mat img = cv::Mat(cv::Size(cols, rows), CV_8UC3, &my_arr);

      std::vector<cv::Rect> face_rects;
      detector.Detect(img, face_rects);
      if (face_rects.empty())
      {
        // No faces
        return PyList_New(0);
      }
      PyObject* face_list = PyList_New(face_rects.size());
      for (int i = 0; i < (int)face_rects.size(); ++i)
      {
        cv::Point point = face_rects[i].tl();
        cv::Size dims = face_rects[i].size();
        PyObject* face_rect = PyTuple_New(4);
        PyTuple_SetItem(face_rect, 0, Py_BuildValue("i", point.x));
        PyTuple_SetItem(face_rect, 1, Py_BuildValue("i", point.y));
        PyTuple_SetItem(face_rect, 2, Py_BuildValue("i", dims.width));
        PyTuple_SetItem(face_rect, 3, Py_BuildValue("i", dims.height));
        PyList_SetItem(face_list, i, face_rect);
      }
      return face_list;
      
      std::cout << "yep" << std::endl;
    }
    std::cout << "Error: Problem parsing image path" << std::endl;
    return PyList_New(0);
  }
  std::cout << "Error: Must call cv2gpu.create_face_recognizer!" << std::endl;
  return PyList_New(0);
}

static PyObject* is_cuda_compatible(PyObject* self, PyObject* args)
{
  #ifdef OPENCV_2
    if (cv::gpu::getCudaEnabledDeviceCount())
    {
      Py_RETURN_TRUE;
    }
  #else
    if (cv::cuda::getCudaEnabledDeviceCount())
    {
      Py_RETURN_TRUE;
    }
  #endif
  Py_RETURN_FALSE;
}

static PyMethodDef cv2gpuMethods[] =
{
  {"init_cpu_detector" , init_cpu_detector , METH_VARARGS, "Initializes CPU OpenCV FaceRecognizer object" },
  {"init_gpu_detector" , init_gpu_detector , METH_VARARGS, "Initializes GPU OpenCV FaceRecognizer object" },
  {"is_cuda_compatible", is_cuda_compatible, METH_NOARGS , "Checks for CUDA compatibility"                },
  {"find_faces"        , find_faces        , METH_VARARGS, "Finds faces using initialized FaceRecognizer" },
  {NULL                , NULL              , 0           , NULL                                           }
};

#if PY_VERSION_HEX >= 0x03000000

/* Python 3.x code */

static struct PyModuleDef cv2gpu =
{
  PyModuleDef_HEAD_INIT,
  "cv2gpu", /* name of module */
  NULL,     /* module documentation, may be NULL */
  -1,       /* size of per-interpreter state of the module,
               or -1 if the module keeps state in global variables. */
  cv2gpuMethods
};

PyMODINIT_FUNC
PyInit_cv2gpu(void)
{
  (void) PyModule_Create(&cv2gpu);
}

#else

/* Python 2.x code */

PyMODINIT_FUNC
initcv2gpu(void)
{
  (void) Py_InitModule("cv2gpu", cv2gpuMethods);
}

#endif
