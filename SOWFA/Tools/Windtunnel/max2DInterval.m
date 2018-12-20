function max2DInterval = max2DInterval(maxDist, probabilityInterval, x1Mesh, x2Mesh)
k = 0;
probabilityVolume = 0; 
[row ,col]  = find(max(max(maxDist))==maxDist);
x1MeshT = x1Mesh - x1Mesh(1,col)*ones(size(x1Mesh));
x2MeshT = x2Mesh - x2Mesh(row,1)*ones(size(x2Mesh));

while probabilityVolume < probabilityInterval
    k = k + 1e-3;
    r = sqrt(x2MeshT.^2 + x1MeshT.^2);
    probabilityVolume = trapz(x2Mesh(:,1),trapz(x1Mesh(1,:),maxDist.*(r < k)))/(trapz(x2Mesh(:,1),trapz(x1Mesh(1,:),maxDist)));
    
end

max2DInterval = [x1Mesh(1,col), x2Mesh(row,1), k];
subplot(2,1,1);
surface(x1Mesh,x2Mesh,maxDist.*(r<k))
meshc(x1Mesh,x2Mesh,maxDist.*(r<k));
limitDistInterval = maxDist.*(r<k);

xlabel('x1');
ylabel('x2');
[xCyl, yCyl, zCyl] = cylinder(k);
xCyl = max2DInterval(1)+xCyl;
yCyl = max2DInterval(2)+yCyl;
zCyl = zCyl *0.1*0.0013;
hold on
surf(xCyl, yCyl, zCyl); 
subplot(2,1,2);
surface(x1Mesh,x2Mesh,maxDist)
meshc(x1Mesh,x2Mesh,maxDist);
xlabel('x1');
ylabel('x2');

end