package com.j2cg.train;

import com.j2cg.base.CTCoolGenCore;

public class CYY1S081_TRAIN008 extends CTCoolGenCore {
	protected group_list m_LocalEntity001;
	protected iyy1_list m_LocalEntity002;
	protected read_lb_iyy1_supplied m_LocalEntity003;
	protected g_list_type m_LocalEntity004;

	@Override
	public void initialize(Object... inputs) {
	}

	@Override
	public void execute(Object... outputs) {

		group_list m_Output01 = (group_list) outputs[0];
		g_list_type m_GroupEnt01 = (g_list_type) m_Output01.getGroupEntity(0);
		
		do {
			 m_Output01.setSubscript(m_Output01.getSubscript()+1);
			 moveMember(m_LocalEntity004, m_GroupEnt01);
			 if (m_LocalEntity002.list_direction.equals("P")) {
				 m_LocalEntity001.setSubscript(m_LocalEntity001.getSubscript()-1);
			 }
			 else {
				 	m_LocalEntity001.setSubscript(m_LocalEntity001.getSubscript()+1);
			 }
			 
			 m_LocalEntity003.total_integer = m_LocalEntity003.total_integer -1;
		} 
		while (m_LocalEntity003.total_integer != 0);

	}

}
