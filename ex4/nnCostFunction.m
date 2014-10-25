function [J grad] = nntrain(nn_params,input_layer_size, hidden_layer_size, num_labels,X, y, lambda)
                                
y


Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
temp=1
for i=2:num_labels
 temp=[temp i]
end 
for i=1:m
 Y(i,:)=temp==y(i);
end
Y

a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);

a2 = [ones(m, 1) a2];
z3 = a2 * Theta2';
ht = sigmoid(z3);

J=J+sum(sum((-1.*Y.*log(ht))-((1.-Y).*(log(1.-ht)))));
J=(1/m)*J;
ts1=Theta1(:,2:end).^2;
ts2=Theta2(:,2:end).^2;
sum1=sum(sum(ts1));
sum2=sum(sum(ts2));
J=J+((lambda/(2*m))*(sum1+sum2));

del_3=ht-Y;                   
Theta2;                         
del_2=(del_3*Theta2(:,2:end));
sg=sigmoidGradient(z2);
del_2=del_2.*sg;    

delta_2=del_3'*a2;       
delta_1=del_2'*a1;    


Theta1;
Theta2;
Theta1_grad=(1/m)*delta_1;
Theta2_grad=(1/m)*delta_2;

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
