document.querySelectorAll('.image-wrapper').forEach(wrapper => {
	const images = wrapper.querySelectorAll('img');
	images[0].classList.add('active');
	const coordinatesData = wrapper.dataset.coordinates.split(';').map(coord => {
	  const [x1, y1, x2, y2, src] = coord.split(',').map(item => (isNaN(item) ? item : Number(item)));
	  return { x1, y1, x2, y2, src };
	});
  
	wrapper.addEventListener('mousemove', (event) => {
	  const rect = wrapper.getBoundingClientRect();
	  const scaleX = rect.width / 1920; // Scale factor for width
	  const scaleY = rect.height / 1080; // Scale factor for height
	  const x = event.clientX - rect.left; // Cursor X position within the wrapper
	  const y = event.clientY - rect.top; // Cursor Y position within the wrapper
  
	  let matched = false;
	  coordinatesData.forEach((region, index) => {
		// Scale the coordinates to match the current size of the wrapper
		const scaledX1 = region.x1 * scaleX;
		const scaledY1 = region.y1 * scaleY;
		const scaledX2 = region.x2 * scaleX;
		const scaledY2 = region.y2 * scaleY;
  
		// Check if the cursor is within the scaled region
		if (x >= scaledX1 && x <= scaledX2 && y >= scaledY1 && y <= scaledY2) {
		  activateImage(images, index + 1); // +1 to skip default image
		  matched = true;
		}
	  });
  
	  // If no region matched, show the default image
	  if (!matched) {
		activateImage(images, 0);
	  }
	});
  
	wrapper.addEventListener('mouseleave', () => {
	  // Reset to default image when leaving the wrapper
	  activateImage(images, 0);
	});
  });
  
  function activateImage(images, index) {
	images.forEach((img, i) => {
	  if (i === index) {
		img.classList.add('active');
	  } else {
		img.classList.remove('active');
	  }
	});
  }
  