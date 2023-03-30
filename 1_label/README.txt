workflow for labeling:
1. use KinderMiner (KM) to extract pairs of entities that might have a relationship
2. gather abstracts that contain those pairs of entities
3. NER labeling w/ prodigy
4. RE labeling w/ prodigy

note that this requires a KM backend server to be running and accessible to your machine ( https://github.com/stewart-lab/fast_km )
after labeling is done, continue on to "2_train"