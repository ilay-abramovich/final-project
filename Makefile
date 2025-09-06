build:
	python3 setup.py build_ext --inplace

clean:
	$(RM) -r build *.so *.pyd
