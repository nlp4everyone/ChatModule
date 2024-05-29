# Update system components
DIRECTORY="system_components"
if [ -d $DIRECTORY ]; then
  echo "$DIRECTORY does exist."
fi
rm -rf system_components
git clone https://github.com/nlp4everyone/system_components.git