import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.objdetect.CascadeClassifier;

public class LicensePlateDetection {

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Load the pre-trained cascade classifier for license plate detection
        CascadeClassifier plateCascade = new CascadeClassifier("haarcascade_russian_plate_number.xml");

        // Load the image
        Mat img = Imgcodecs.imread("car_with_plate.jpg");

        // Convert the image to grayscale
        Mat gray = new Mat();
        Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY);

        // Detect license plates in the image
        MatOfRect plates = new MatOfRect();
        plateCascade.detectMultiScale(gray, plates);

        // Draw rectangles around the detected license plates
        for (Rect rect : plates.toArray()) {
            Imgproc.rectangle(img, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(255, 0, 0), 2);
        }

        // Display the result
        Imgcodecs.imwrite("detected_plates.jpg", img);
    }
}
