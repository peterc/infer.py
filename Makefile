VERSION = $(shell grep version pyproject.toml | head -1 | cut -d '"' -f 2)

.PHONY: build clean upload release

clean:
	rm -rf build dist *.egg-info

build: clean
	python -m build

upload:
	twine upload dist/*

release: build upload
	git add pyproject.toml CHANGELOG.md
	git commit -m "Release version $(VERSION)"
	git tag v$(VERSION)
	git push origin v$(VERSION)
