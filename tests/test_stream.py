import unittest

from peleng.stream import SequentialVideo, due_frame


class FakeCapture:
    def __init__(self):
        self.index = -1
        self.grabs = []

    def grab(self):
        self.index += 1
        self.grabs.append(self.index)
        return self.index < 10

    def retrieve(self):
        return True, self.index

    def release(self):
        pass


class StreamTests(unittest.TestCase):
    def test_clock_never_releases_a_future_frame(self):
        self.assertEqual(due_frame(0.099, 30, 100), 2)
        self.assertEqual(due_frame(0, 30, 100), 0)
        self.assertEqual(due_frame(10, 30, 100), 99)

    def test_reader_discards_old_frames_without_reading_past_deadline(self):
        capture = FakeCapture()
        reader = SequentialVideo(capture)
        self.assertEqual(reader.read_to(3), (3, 3))
        self.assertEqual(capture.grabs, [0, 1, 2, 3])
        self.assertEqual(reader.read_to(3), (3, 3))
        self.assertEqual(reader.read_to(5), (5, 5))
        self.assertEqual(capture.grabs, list(range(6)))

    def test_reader_rejects_backward_access(self):
        reader = SequentialVideo(FakeCapture())
        reader.read_to(4)
        with self.assertRaises(ValueError):
            reader.read_to(2)


if __name__ == '__main__':
    unittest.main()
