package com.j2cg.train;

import com.j2cg.base.CTBreakException;
import com.j2cg.base.CTCoolGenCore;
import com.j2cg.base.CTCoolGenGlobal;
import com.j2cg.base.CTDBEntityQuery;
import com.j2cg.base.CTDBException;

public class DYYY0151_TRAIN009 extends CTCoolGenCore {
	protected parent m_LocalEntity001;
	protected from_parent m_InputEntity002;
	protected filter_start_parent m_InputEntity003;
	protected filter_stop_parent m_InputEntity004;
	protected filter_parent m_InputEntity005;
	protected filter_iyy1_list m_InputEntity006;
	protected iyy1_component m_InputEntity007;

	@Override
	public void initialize(Object... inputs) {
		setDBConnection(inputs);
	}

	@Override
	public void execute(Object... outputs) {
		
		filter_iyy1_list m_LocalEntity002 = new filter_iyy1_list();
		dont_change_reason_codes m_LocalEntity003 = new dont_change_reason_codes();
		
		group_list m_Output01 = (group_list) outputs[0];
		error_iyy1_component m_Output02 = (error_iyy1_component) outputs[1];
		
		try {
				CTDBEntityQuery query = new CTDBEntityQuery(m_DBConnection, parent.class.getName());
				query.addOrderField("pkey_attr_text", true);
				query.addCondition(null, "pkey_attr_text", "<=", m_InputEntity002.pkey_attr_text);
				query.addCondition("AND", "pkey_attr_text", ">=", m_InputEntity003.pkey_attr_text);
				query.addCondition("AND", "pkey_attr_text", "<=", m_InputEntity004.pkey_attr_text);
				query.addCondition("AND", "pkey_attr_text", "IS LIKE", m_InputEntity005.pkey_attr_text);
				query.run();
				
				while (query.next()) {
					m_LocalEntity001 = (parent) query.fetchEntity();
					if (m_Output01.getSubscript() < m_InputEntity006.scroll_amount) {
						if (m_LocalEntity002.scroll_type.equals("S")) {
							m_LocalEntity002.scroll_type = CTCoolGenGlobal.SPACES; 
						}
						else {
								m_Output01.setSubscript(m_Output01.getSubscript()+1);
								moveMember(m_LocalEntity001, m_Output01.getGroupEntity(0));
						}
					}
					else {
							m_Output02.reason_code = m_LocalEntity003.m_11_list_full;
							throw new CTBreakException();
					}
				}
				
		} catch (CTBreakException be) {
			
		} catch (CTDBException de) {
			
		}
	}

}
