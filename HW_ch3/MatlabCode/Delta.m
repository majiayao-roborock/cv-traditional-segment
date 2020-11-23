
function DPhi = Delta( phi , epsilon );
	DPhi = (epsilon/pi)./(epsilon^2+ phi.^2);
end