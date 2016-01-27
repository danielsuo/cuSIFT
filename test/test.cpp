#include "gtest/gtest.h"

#include <iostream>
#include <cmath>
#include <iomanip>
#include <string>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cuImage.h"
#include "cuSIFT.h"

#include "extras/matching.h"
#include "extras/homography.h"
#include "extras/debug.h"

// IndependentMethod is a test case - here, we have 2 tests for this 1 test case
TEST(IndependentMethod, ResetsToZero) {
  EXPECT_EQ(0, 0);
  ASSERT_EQ(12, 12);
}
