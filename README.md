# IndoorVG

This IndoorVG dataset is a Scene Graph Generation dataset for Service Robotics, introduced in the paper [In Defense of Scene Graph Generation for Human-Robot Open-Ended Interaction in Service Robotics](https://link.springer.com/chapter/10.1007/978-3-031-55015-7_25). It is a split of Visual Genome targeting real-world applications in indoor settings. The IndoorVG dataset is composed of 83 object classes and 34 predicate classes which have been manually selected and refined using different semi-automatic merging and processing techniques. To use it you can download the VG images from the following links: [Part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [Part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). The annotated scene graphs can be downloaded [from this link](https://drive.google.com/file/d/1zfKXzmLxxYMzwlECtSch84oknCBEXTzI/view?usp=sharing). The original split is VG-SGG.h5, then the augmented split with penet adds up to 34% more annotations and refines existing ones. We refer to the annotations from ```VG-SGG-augmented-penet.h5``` as the ```IndoorVG_V4``` dataset.
